"""Clean-v6 三卡（A/B/C）全自动后台实验与评测守护脚本。

特点：
1. 幂等断点续跑：任何训练或评测步骤若目标产物（.pt + .json 或评测 .json）已存在且非空，自动跳过。
2. 跨卡依赖自动等待：B 卡联合训练自动等待 A 卡对应 seed 的 A3 Tower 权重；C 卡在等待 B2 联合权重期间先行跑 Teacher / Base / LLaVA-Proj / Pool32 / FastV32 统一协议训练与评测。
3. 严格执行方案全部要求：
   - 阶段 1：GPU 门禁 verify_foundation_gate.py
   - 阶段 2-A：Tower 架构消融 A0 / A1(no_rot) / A2 / A3（2400 步，gnd=0，seed 0/1/2）+ A3 gnd=0.35 对照
   - 阶段 2-B：无塔前端公平对照 B0(K=0) / B1(LoRA K=4) / B2(rots K=4)（统一 2200 步预热 + 1600 步联合训练，gnd=0，seed 0/1/2）+ B2 gnd=0.35 对照
   - 阶段 2-C：原始 LLM 100% 冻结的 OPD 四组对照 C0(none) / C1(offkd) / C2(fwd) / C3(jsd)（600 步，bs=4，seed 0/1/2）
   - 阶段 3：6 公开榜统一无禁词评测（真图 none、黑图 blank、3 次独立随机图像组异图置换 seed=101/102/103、强制自由生成 force-gen）、独立进程真实效率测试（bench_efficiency.py）与纯文本保持测试（eval_lang.py）。
"""

import argparse
import os
import subprocess
import sys
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
    line = f"[{ts}][GPU-{gpu}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"pipeline_gpu{gpu}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 100


def wait_for_file(gpu: str, p: Path, poll_sec: int = 20):
    while not file_ok(p):
        log(gpu, f"Waiting for prerequisite file: {p} ...")
        time.sleep(poll_sec)
    log(gpu, f"Prerequisite ready: {p}")


def run_cmd(gpu: str, tag: str, cmd: list, done_files: list):
    if all(file_ok(Path(p)) for p in done_files):
        log(gpu, f"[SKIP] {tag} already completed -> {[str(p) for p in done_files]}")
        return True
    log(gpu, f"[START] {tag}: {' '.join(str(x) for x in cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    res = subprocess.run([str(x) for x in cmd], cwd=str(PROJ), env=env)
    dt = time.time() - t0
    if res.returncode != 0:
        log(gpu, f"[FAIL] {tag} exited with rc={res.returncode} after {dt:.1f}s")
        return False
    log(gpu, f"[DONE] {tag} finished in {dt:.1f}s")
    return True


def run_eval_suite(gpu: str, mode: str, ckpt: str, base_name: str, full_controls: bool = False, with_fgen: bool = False):
    """评测公开榜：
    - 默认跑 6 榜真图 (none) + 3 榜异图置换 seed=101
    - full_controls=True 时额外跑 6 榜黑图 (blank) + 异图置换 seed=102, 103
    - with_fgen=True 时额外跑 MMStar/Hallusion/MathVista 强制贪心自由生成 (--force-gen)
    """
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    ckpt_args = ["--ckpt", ckpt] if ckpt else []

    # 1. 6 榜真图
    run_cmd(
        gpu,
        f"eval:{base_name}:real",
        [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", base_name,
         "--image-ctrl", "none", "--out-dir", EVAL_OUT],
        [EVAL_OUT / f"{base_name}.json"],
    )

    # 2. 异图置换 seed=101 (Core3: mmstar,hallusion,mathvista)
    s101_name = f"{base_name}_shuf101"
    run_cmd(
        gpu,
        f"eval:{s101_name}",
        [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", s101_name,
         "--benches", "mmstar,hallusion,mathvista", "--image-ctrl", "shuffle", "--shuffle-seed", "101",
         "--out-dir", EVAL_OUT],
        [EVAL_OUT / f"{s101_name}.json"],
    )

    if full_controls:
        # 3. 异图置换 seed=102, 103
        for sd in (102, 103):
            sname = f"{base_name}_shuf{sd}"
            run_cmd(
                gpu,
                f"eval:{sname}",
                [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", sname,
                 "--benches", "mmstar,hallusion,mathvista", "--image-ctrl", "shuffle", "--shuffle-seed", str(sd),
                 "--out-dir", EVAL_OUT],
                [EVAL_OUT / f"{sname}.json"],
            )
        # 4. 6 榜全黑图对照
        bname = f"{base_name}_blank"
        run_cmd(
            gpu,
            f"eval:{bname}",
            [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", bname,
             "--image-ctrl", "blank", "--out-dir", EVAL_OUT],
            [EVAL_OUT / f"{bname}.json"],
        )

    if with_fgen:
        fname = f"{base_name}_fgen"
        run_cmd(
            gpu,
            f"eval:{fname}",
            [PY, SCRIPTS / "eval_benchmarks.py", "--mode", mode, *ckpt_args, "--name", fname,
             "--benches", "mmstar,hallusion,mathvista", "--force-gen", "--out-dir", EVAL_OUT],
            [EVAL_OUT / f"{fname}.json"],
        )


def train_tower_variant(gpu: str, tag: str, extra_flags: list, seed: int, gnd_w: float = 0.0):
    prefix = OUT / f"{tag}_s{seed}"
    pt_file = OUT / f"{tag}_s{seed}_pt.pt"
    json_file = OUT / f"{tag}_s{seed}.json"
    log_file = OUT / f"{tag}_s{seed}.log"
    cmd = [
        PY, SCRIPTS / "train_final.py",
        "--arm", "ce",
        "--data", DATA_SPEC,
        "--steps", "2400",
        "--bs", "8",
        "--seq", "256",
        "--lr", "5e-4",
        "--eval-every", "800",
        "--em-n", "1000",
        "--gnd-weight", str(gnd_w),
        "--gnd-margin", "0.75",
        "--seed", str(seed),
        *extra_flags,
        "--result-json", str(json_file),
        "--log", str(log_file),
    ]
    run_cmd(gpu, f"train:{prefix.name}", cmd, [pt_file, json_file])
    return pt_file


def pipeline_gpu_a():
    gpu = "A"
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    # 0. 确保实验底座门禁完成
    gate_json = OUT / "foundation_gate_manifest.json"
    while not file_ok(gate_json):
        # 检查是否已有 verify_foundation_gate.py 正在运行
        ps = subprocess.run(["pgrep", "-f", "verify_foundation_gate.py"], capture_output=True, text=True)
        if ps.returncode == 0 and ps.stdout.strip():
            log(gpu, f"verify_foundation_gate.py is currently running (PID={ps.stdout.strip()}), waiting...")
            time.sleep(15)
        else:
            run_cmd(gpu, "foundation_gate", [PY, SCRIPTS / "verify_foundation_gate.py"], [gate_json])
            break

    # 1. 优先跑 A3 (Full SpinVL-Tower) seed 0，供 GPU-B 联合训练立刻使用！
    a3_flags = ["--multifreq", "--hybrid", "--qcond", "--direct-hyb"]
    a2_flags = ["--hybrid", "--qcond", "--direct-hyb"]
    a1_flags = ["--no-rot", "--hybrid", "--direct-hyb"]
    a0_flags = []

    pt_a3_s0 = train_tower_variant(gpu, "v6_tower_A3", a3_flags, seed=0, gnd_w=0.0)
    # 紧接着跑 A3 seed 1, 2 的训练，让 GPU-B 的 seed 1, 2 也完全不用等
    pt_a3_s1 = train_tower_variant(gpu, "v6_tower_A3", a3_flags, seed=1, gnd_w=0.0)
    pt_a3_s2 = train_tower_variant(gpu, "v6_tower_A3", a3_flags, seed=2, gnd_w=0.0)

    # 2. 跑 A0 (ROSE), A1 (CA-only no_rot), A2 (Single-freq + qcond + CA) 的 seed 0
    pt_a0_s0 = train_tower_variant(gpu, "v6_tower_A0_rose", a0_flags, seed=0, gnd_w=0.0)
    pt_a1_s0 = train_tower_variant(gpu, "v6_tower_A1_caonly", a1_flags, seed=0, gnd_w=0.0)
    pt_a2_s0 = train_tower_variant(gpu, "v6_tower_A2_single", a2_flags, seed=0, gnd_w=0.0)

    # 3. 评测 seed 0 全套 Tower 消融（A3 跑全套对照 + 自由生成；A0 跑全套对照；A1/A2 跑真图+异图）
    run_eval_suite(gpu, "tower", str(pt_a3_s0), "v6_tower_A3_s0", full_controls=True, with_fgen=True)
    run_eval_suite(gpu, "tower", str(pt_a0_s0), "v6_tower_A0_rose_s0", full_controls=True, with_fgen=True)
    run_eval_suite(gpu, "tower", str(pt_a1_s0), "v6_tower_A1_caonly_s0", full_controls=False, with_fgen=False)
    run_eval_suite(gpu, "tower", str(pt_a2_s0), "v6_tower_A2_single_s0", full_controls=False, with_fgen=False)

    # 4. 评测 A3 seed 1, 2
    run_eval_suite(gpu, "tower", str(pt_a3_s1), "v6_tower_A3_s1", full_controls=False, with_fgen=False)
    run_eval_suite(gpu, "tower", str(pt_a3_s2), "v6_tower_A3_s2", full_controls=False, with_fgen=False)

    # 5. 第 3.4 节：接地辅助损失对照（Tower A3 gnd=0.35, seed=0）
    pt_a3_gnd = train_tower_variant(gpu, "v6_tower_A3_gnd035", a3_flags, seed=0, gnd_w=0.35)
    run_eval_suite(gpu, "tower", str(pt_a3_gnd), "v6_tower_A3_gnd035_s0", full_controls=True, with_fgen=False)

    # 6. 跑 A0, A1, A2 的 seed 1, 2 并评测
    for sd in (1, 2):
        pt0 = train_tower_variant(gpu, "v6_tower_A0_rose", a0_flags, seed=sd, gnd_w=0.0)
        run_eval_suite(gpu, "tower", str(pt0), f"v6_tower_A0_rose_s{sd}", full_controls=False, with_fgen=False)
        pt1 = train_tower_variant(gpu, "v6_tower_A1_caonly", a1_flags, seed=sd, gnd_w=0.0)
        run_eval_suite(gpu, "tower", str(pt1), f"v6_tower_A1_caonly_s{sd}", full_controls=False, with_fgen=False)
        pt2 = train_tower_variant(gpu, "v6_tower_A2_single", a2_flags, seed=sd, gnd_w=0.0)
        run_eval_suite(gpu, "tower", str(pt2), f"v6_tower_A2_single_s{sd}", full_controls=False, with_fgen=False)

    log(gpu, "ALL GPU-A TASKS COMPLETED SUCCESSFULLY!")


def train_free_variant(gpu: str, tag: str, fmode: str, K: int, seed: int, gnd_w: float = 0.0):
    # Stage 1: 统一 2200 步前端特征对齐预热
    front_out = OUT / f"{tag}_front_s{seed}"
    front_pt = OUT / f"{tag}_front_s{seed}.pt"
    front_json = OUT / f"{tag}_front_s{seed}.json"
    cmd_front = [
        PY, SCRIPTS / "train_front.py",
        "--mode", fmode,
        "--K", str(K),
        "--lora-rank", "16",
        *SRC_ARGS,
        "--bs", "16",
        "--steps", "2200",
        "--lr", "3e-4",
        "--lr-static", "1e-3",
        "--eval-every", "550",
        "--n-val", "400",
        "--seed", str(seed),
        "--out", str(front_out),
    ]
    run_cmd(gpu, f"train_front:{front_out.name}", cmd_front, [front_pt, front_json])

    # Stage 2: 统一 1600 步联合训练（等待 GPU-A 产出同 seed 的 A3 Tower 权重）
    vrmg_ckpt = OUT / f"v6_tower_A3_s{seed}_pt.pt"
    wait_for_file(gpu, vrmg_ckpt)

    joint_out = OUT / f"{tag}_joint_s{seed}"
    joint_pt = OUT / f"{tag}_joint_s{seed}.pt"
    joint_json = OUT / f"{tag}_joint_s{seed}.json"
    cmd_joint = [
        PY, SCRIPTS / "train_joint.py",
        "--stage", "joint",
        "--data", DATA_SPEC,
        *SRC_ARGS,
        "--vrmg-ckpt", str(vrmg_ckpt),
        "--front-ckpt", str(front_pt),
        "--front-mode", fmode,
        "--K", str(K),
        "--lora-rank", "16",
        "--multifreq", "--hybrid", "--qcond", "--direct-hyb",
        "--arm", "ce",
        "--bs", "8",
        "--seq", "256",
        "--steps", "1600",
        "--lr-vrmg", "2e-4",
        "--lr-front", "1e-4",
        "--feat-weight", "1.0",
        "--gnd-weight", str(gnd_w),
        "--gnd-margin", "0.75",
        "--n-eval", "600",
        "--n-em", "1000",
        "--seed", str(seed),
        "--out", str(joint_out),
    ]
    run_cmd(gpu, f"train_joint:{joint_out.name}", cmd_joint, [joint_pt, joint_json])
    return front_pt, joint_pt


def pipeline_gpu_b():
    gpu = "B"
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    # 1. 优先跑 B2 (rots K=4) seed 0，让 GPU-C 的 OPD 四组实验第一时间拿到起点！
    _, pt_b2_s0 = train_free_variant(gpu, "v6_free_B2_rots_K4", "rots", K=4, seed=0, gnd_w=0.0)
    run_eval_suite(gpu, "tower_free", str(pt_b2_s0), "v6_free_B2_rots_K4_s0", full_controls=True, with_fgen=True)

    # 2. 跑 B0 (none K=0) 与 B1 (lora K=4) 的 seed 0（统一 2200 步预热 + 1600 步联合训练！）
    _, pt_b0_s0 = train_free_variant(gpu, "v6_free_B0_none_K0", "none", K=0, seed=0, gnd_w=0.0)
    run_eval_suite(gpu, "tower_free", str(pt_b0_s0), "v6_free_B0_none_K0_s0", full_controls=True, with_fgen=False)

    _, pt_b1_s0 = train_free_variant(gpu, "v6_free_B1_lora_K4", "lora", K=4, seed=0, gnd_w=0.0)
    run_eval_suite(gpu, "tower_free", str(pt_b1_s0), "v6_free_B1_lora_K4_s0", full_controls=True, with_fgen=False)

    # 3. 跑 B2 (rots K=4) 的 seed 1, 2，供 GPU-C 的多 seed OPD 使用
    for sd in (1, 2):
        _, pt_b2_sd = train_free_variant(gpu, "v6_free_B2_rots_K4", "rots", K=4, seed=sd, gnd_w=0.0)
        run_eval_suite(gpu, "tower_free", str(pt_b2_sd), f"v6_free_B2_rots_K4_s{sd}", full_controls=False, with_fgen=False)

    # 4. 第 3.4 节：无塔 B2 gnd=0.35 对照 (seed 0)
    _, pt_b2_gnd = train_free_variant(gpu, "v6_free_B2_rots_K4_gnd035", "rots", K=4, seed=0, gnd_w=0.35)
    run_eval_suite(gpu, "tower_free", str(pt_b2_gnd), "v6_free_B2_rots_K4_gnd035_s0", full_controls=True, with_fgen=False)

    # 5. 跑 B0, B1 的 seed 1, 2 并评测
    for sd in (1, 2):
        _, pt_b0_sd = train_free_variant(gpu, "v6_free_B0_none_K0", "none", K=0, seed=sd, gnd_w=0.0)
        run_eval_suite(gpu, "tower_free", str(pt_b0_sd), f"v6_free_B0_none_K0_s{sd}", full_controls=False, with_fgen=False)
        _, pt_b1_sd = train_free_variant(gpu, "v6_free_B1_lora_K4", "lora", K=4, seed=sd, gnd_w=0.0)
        run_eval_suite(gpu, "tower_free", str(pt_b1_sd), f"v6_free_B1_lora_K4_s{sd}", full_controls=False, with_fgen=False)

    log(gpu, "ALL GPU-B TASKS COMPLETED SUCCESSFULLY!")


def train_opd_variant(gpu: str, tag: str, kl_mode: str, kl_w: float, seed: int):
    joint_ckpt = OUT / f"v6_free_B2_rots_K4_joint_s{seed}.pt"
    wait_for_file(gpu, joint_ckpt)
    opd_out = OUT / f"{tag}_s{seed}"
    opd_pt = OUT / f"{tag}_s{seed}.pt"
    opd_json = OUT / f"{tag}_s{seed}.json"
    cmd = [
        PY, SCRIPTS / "train_joint_opd.py",
        "--data", DATA_SPEC,
        *SRC_ARGS,
        "--joint-ckpt", str(joint_ckpt),
        "--multifreq", "--hybrid", "--qcond", "--direct-hyb",
        "--steps", "600",
        "--bs", "4",
        "--accum", "1",
        "--max-new", "24",
        "--min-new", "1",
        "--gen-temp", "0.7",
        "--kd-temp", "1.0",
        "--kl-mode", kl_mode,
        "--kl-weight", str(kl_w),
        "--ce-weight", "1.0",
        "--feat-weight", "0.2",
        "--lr-vrmg", "2e-5",
        "--lr-front", "1e-5",
        "--n-eval", "600",
        "--n-acc", "150",
        "--n-em", "1000",
        "--seed", str(seed),
        "--out", str(opd_out),
    ]
    run_cmd(gpu, f"train_opd:{opd_out.name}", cmd, [opd_pt, opd_json])
    return opd_pt


def pipeline_gpu_c():
    gpu = "C"
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL_OUT.mkdir(parents=True, exist_ok=True)

    # 1. 在等待 GPU-B 产出 B2_s0 期间，先行完成无图基座 Base、Teacher 与 LLaVA-Proj 基线的统一无禁词协议训练与评测
    run_eval_suite(gpu, "base", "", "v6_base_qwen3_17b", full_controls=False, with_fgen=True)
    run_eval_suite(gpu, "teacher", "", "v6_teacher_qwen3vl8b", full_controls=True, with_fgen=False)

    # 训练 44k 同划分 LLaVA-Proj 基线 (2400 步, gnd=0.0, seed=0)
    proj_out = OUT / "v6_proj_llava_s0"
    proj_pt = OUT / "v6_proj_llava_s0.pt"
    proj_json = OUT / "v6_proj_llava_s0.json"
    run_cmd(
        gpu,
        "train:v6_proj_llava_s0",
        [PY, SCRIPTS / "train_proj.py", "--data", DATA_SPEC, *SRC_ARGS,
         "--steps", "2400", "--bs", "8", "--seq", "256", "--lr", "5e-4",
         "--fast-feats", "--gnd-weight", "0.0", "--seed", "0", "--out", str(proj_out)],
        [proj_pt, proj_json],
    )
    run_eval_suite(gpu, "proj", str(proj_pt), "v6_proj_llava_s0", full_controls=True, with_fgen=True)
    run_eval_suite(gpu, "proj_pool32", str(proj_pt), "v6_proj_pool32_s0", full_controls=False, with_fgen=False)
    run_eval_suite(gpu, "proj_fastv32", str(proj_pt), "v6_proj_fastv32_s0", full_controls=False, with_fgen=False)

    # 2. 第 3.3 节：原始 LLM 100% 冻结下的 OPD 四组干净对照 (C0/C1/C2/C3, seed=0)
    opd_specs = [
        ("v6_opd_C0_cont_ce", "none", 0.0, False),
        ("v6_opd_C1_offkd", "offkd", 0.25, False),
        ("v6_opd_C2_fwdkl", "fwd", 0.25, True),
        ("v6_opd_C3_jsd", "jsd", 0.25, True),
    ]
    for tag, kl_m, kl_w, full_ctrl in opd_specs:
        pt_c = train_opd_variant(gpu, tag, kl_m, kl_w, seed=0)
        run_eval_suite(gpu, "tower_free", str(pt_c), f"{tag}_s0", full_controls=full_ctrl, with_fgen=True)

    # 3. 第 3.5 & 3.6 节：独立子进程效率基准（prefill/decode 实测延迟、显存与解析 FLOPs）与纯文本能力保持评测
    b2_joint_s0 = OUT / "v6_free_B2_rots_K4_joint_s0.pt"
    a3_tower_s0 = OUT / "v6_tower_A3_s0_pt.pt"
    a0_rose_s0 = OUT / "v6_tower_A0_rose_s0_pt.pt"
    b1_front_s0 = OUT / "v6_free_B1_lora_K4_front_s0.pt"
    b2_front_s0 = OUT / "v6_free_B2_rots_K4_front_s0.pt"

    wait_for_file(gpu, b2_joint_s0)
    wait_for_file(gpu, a3_tower_s0)
    wait_for_file(gpu, a0_rose_s0)
    eff_json = OUT / "v6_efficiency_K4.json"
    run_cmd(
        gpu,
        "bench_efficiency",
        [PY, SCRIPTS / "bench_efficiency.py",
         "--modes", "ours,tower_ours,rose,keep,llava",
         "--joint-ckpt", str(b2_joint_s0),
         "--tower-ckpt", str(a3_tower_s0),
         "--rose-ckpt", str(a0_rose_s0),
         "--proj-ckpt", str(proj_pt),
         "--out", str(eff_json)],
        [eff_json],
    )

    wait_for_file(gpu, b1_front_s0)
    wait_for_file(gpu, b2_front_s0)
    lang_json = OUT / "v6_lang_retention.json"
    run_cmd(
        gpu,
        "eval_lang",
        [PY, SCRIPTS / "eval_lang.py",
         "--variant", f"lora4=lora:{b1_front_s0}",
         "--variant", f"rots4=rots:{b2_front_s0}",
         "--out", str(lang_json)],
        [lang_json],
    )

    # 4. 跑 C0/C1/C2/C3 的 seed 1, 2 并评测
    for sd in (1, 2):
        for tag, kl_m, kl_w, _ in opd_specs:
            pt_c_sd = train_opd_variant(gpu, tag, kl_m, kl_w, seed=sd)
            run_eval_suite(gpu, "tower_free", str(pt_c_sd), f"{tag}_s{sd}", full_controls=False, with_fgen=True)

    log(gpu, "ALL GPU-C TASKS COMPLETED SUCCESSFULLY!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", choices=["A", "B", "C"], required=True)
    args = ap.parse_args()
    if args.gpu == "A":
        pipeline_gpu_a()
    elif args.gpu == "B":
        pipeline_gpu_b()
    else:
        pipeline_gpu_c()


if __name__ == "__main__":
    main()
