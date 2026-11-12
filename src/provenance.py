"""统一来源追踪（Provenance）、原始 LLM 冻结硬门禁校验、以及跨阶段统一数据切分入口。"""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

TRACKED_REL_FILES = (
    "src/vrmg.py",
    "src/front.py",
    "src/wrap.py",
    "src/imgdata.py",
    "src/provenance.py",
    "scripts/bench_data.py",
    "scripts/bench_models.py",
    "scripts/bench_score.py",
    "scripts/eval_benchmarks.py",
    "scripts/train_final.py",
    "scripts/train_front.py",
    "scripts/train_joint.py",
    "scripts/train_opd.py",
    "scripts/train_joint_opd.py",
    "scripts/bench_efficiency.py",
)

PROMPT_PROTOCOL_VERSION = "clean_v6_unified_greedy_max24_no_ban_v2"


def compute_code_hashes(root: Path = ROOT) -> dict:
    out = {}
    for rel in TRACKED_REL_FILES:
        p = root / rel
        if p.exists():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def fast_file_sha256(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def snapshot_backbone_hashes(model: torch.nn.Module):
    """对原始 LLM 的全部参数张量计算逐张量 SHA1（自动剥离 RotatedProj 的 .rot.base. 包装路径）。"""
    per_tensor = {}
    agg = hashlib.sha1()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0].replace(".rot.base.", ".")):
        clean_name = name.replace(".rot.base.", ".")
        raw = p.detach().cpu().float().numpy().tobytes()
        sh = hashlib.sha1(raw).hexdigest()
        per_tensor[clean_name] = sh
        agg.update(f"{clean_name}:{sh};".encode("utf-8"))
    return per_tensor, agg.hexdigest()


def verify_frozen_backbone_and_optimizer(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer = None,
    allowed_params=None,
    pre_hashes: dict = None,
) -> dict:
    """硬性断言：
    1. model.parameters() 中没有任何 requires_grad=True 的参数；
    2. optimizer 参数集合必须 100% 属于 allowed_params 白名单，且与 model.parameters() 交集为空；
    3. 若提供训练前快照 pre_hashes，则逐张量校验全部原始 LLM 参数 SHA1 零变化。
    """
    req_grad_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert not req_grad_names, f"冻结违规：原始 LLM 存在 requires_grad=True 参数: {req_grad_names[:5]}"

    model_param_ids = {id(p) for p in model.parameters()}
    if opt is not None:
        opt_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        overlap = opt_param_ids & model_param_ids
        assert len(overlap) == 0, f"冻结违规：Optimizer 混入了 {len(overlap)} 个原始 LLM 参数！"
        if allowed_params is not None:
            allowed_ids = {id(p) for p in allowed_params}
            extra = opt_param_ids - allowed_ids
            assert len(extra) == 0, f"冻结违规：Optimizer 存在 {len(extra)} 个非白名单参数！"

    changed = []
    post_agg_sha1 = ""
    if pre_hashes is not None:
        post_hashes, post_agg_sha1 = snapshot_backbone_hashes(model)
        assert set(post_hashes.keys()) == set(pre_hashes.keys()), "训练前后原始 LLM 参数张量名称集合不一致！"
        for k, sh_before in pre_hashes.items():
            if post_hashes[k] != sh_before:
                changed.append(k)
        assert not changed, f"冻结违规：训练后有 {len(changed)} 个原始 LLM 张量发生变化: {changed[:5]}"

    return {
        "frozen_verified": True,
        "backbone_tensor_count": len(pre_hashes) if pre_hashes is not None else len(model_param_ids),
        "changed_backbone_tensors": len(changed),
        "backbone_sha1": post_agg_sha1,
    }


def build_run_manifest(stage: str, args_dict: dict, split_info: dict = None, extra: dict = None) -> dict:
    try:
        import transformers
        tf_ver = transformers.__version__
    except Exception:
        tf_ver = "unknown"
    try:
        import bench_score as bs
        scorer_ver = bs.SCORER_VERSION
    except Exception:
        scorer_ver = "clean_v6_official_relaxed_v2"

    ckpt_hashes = {}
    for k in ("vrmg_ckpt", "front_ckpt", "joint_ckpt", "init_ckpt", "ckpt"):
        v = args_dict.get(k)
        if isinstance(v, str) and v and os.path.exists(v):
            ckpt_hashes[k] = fast_file_sha256(v)

    manifest = {
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_protocol_version": PROMPT_PROTOCOL_VERSION,
        "scorer_version": scorer_ver,
        "code_sha256": compute_code_hashes(),
        "ckpt_sha256": ckpt_hashes,
        "split_info": split_info or {},
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": tf_ver,
            "cuda": torch.version.cuda or "none",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }
    if extra:
        manifest.update(extra)
    return manifest
