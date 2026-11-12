"""叙事 2 · Stage 2A：把 8B 老师的视觉塔蒸进学生前 K 层（内置视觉编码器）。

arm（--mode）：rot 静态正交旋转｜lora 加性 LoRA（VoRA 式对照）｜full 前 K 层整层复制可训（上限）｜none 且 K=0（对照：无 LLM 层）
"""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
from front import PatchFront, feature_loss  # noqa: E402
from provenance import capture_run_inputs, build_run_manifest, snapshot_backbone_hashes, verify_frozen_backbone_and_optimizer  # noqa: E402
from train_final import load_data, split_train_val_by_image as tf_split_by_image  # noqa: E402
from wrap import attach_rotation, freeze_all_but  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"


def prepare_front_split(source_specs, seed=0):
    """前端预热的数据加载与图像级切分入口（供训练与门禁脚本直接调用验证跨阶段一致性）。"""
    meta_list, paths = [], {}
    for spec in source_specs:
        m_p, p_map = imgdata.parse_source(spec)
        meta_list.append(m_p)
        paths.update(p_map)
    data = load_data(",".join(meta_list))
    train_idx, val_idx, split_info = tf_split_by_image(data, paths, seed=seed)
    train_items = [(i, paths[data["ids"][i]]) for i in train_idx if data["ids"][i] in paths and data["vlen"][i] > 0]
    val_items = [(i, paths[data["ids"][i]]) for i in val_idx if data["ids"][i] in paths and data["vlen"][i] > 0]
    return data, paths, train_idx, val_idx, train_items, val_items, split_info


@torch.no_grad()
def evaluate(front, loader, dev, max_batches=40):
    front.eval()
    cos, n = 0.0, 0
    for bi, (pv, grid, lens, tgts) in enumerate(loader):
        if bi >= max_batches:
            break
        v_hat, cl = front(pv.to(dev), grid, lens)
        _, c = feature_loss(v_hat, tgts, cl)
        cos += c.item() * len(lens)
        n += len(lens)
    front.train()
    return cos / max(n, 1)  # 平均余弦相似度（相同特征为 1.0，正交为 0.0）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rot", "rots", "lora", "full", "none"], default="rot")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--source", action="append", required=True, help="meta|manifest|img_root，可重复")
    ap.add_argument("--eval-source", action="append", default=[])
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--teacher", default=imgdata.TEACHER)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积：每个优化步用 accum 个 bs 大小的微批")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-static", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="输出前缀，生成 <out>.log/.json/.pt")
    args = ap.parse_args()
    capture_run_inputs(vars(args))
    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(args.out + ".log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    ip = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True).image_processor
    data, paths, train_idx, val_idx, train_items, val_items, split_info = prepare_front_split(args.source, seed=args.seed)
    if args.n_val and args.n_val < len(val_items):
        val_eval_items = val_items[: args.n_val]
    else:
        val_eval_items = val_items
    log(f"=== front mode={args.mode} K={args.K} train={len(train_items)} val={len(val_items)}(eval={len(val_eval_items)}) "
        f"split={split_info} ===")

    train_ds = imgdata.FrontDataset(train_items, data["vfeat"], ip)
    val_ds = imgdata.FrontDataset(val_eval_items, data["vfeat"], ip)
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers,
                          collate_fn=imgdata.collate, drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=imgdata.collate)

    eval_lds = {}
    for k, spec in enumerate(args.eval_source):
        meta, paths = imgdata.parse_source(spec)
        d = load_data(meta)
        its = [(i, paths[r]) for i, r in enumerate(d["ids"]) if r in paths][:400]
        eval_lds[os.path.basename(meta)] = DataLoader(imgdata.FrontDataset(its, d["vfeat"], ip), batch_size=args.bs,
                                                      num_workers=2, collate_fn=imgdata.collate)

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    K = 0 if args.mode == "none" else args.K
    attach_rotation(model, [], wrap_extra=list(range(K)))
    vdim = train_ds[0][2].shape[-1]
    front = PatchFront(model, K=K, mode=args.mode, vdim=vdim, lora_rank=args.lora_rank).to(dev)
    params = front.trainable_params()
    freeze_all_but(model, [])
    for p in params:
        p.requires_grad_(True)
    n_tr = sum(p.numel() for p in params)
    n_static = sum(p.numel() for p in (front.static.parameters() if front.static is not None else []))
    n_merger = sum(p.numel() for p in front.merger.parameters())
    n_patch = sum(p.numel() for p in front.patch_embed.parameters()) + \
              sum(p.numel() for p in front.row_emb.parameters()) + \
              sum(p.numel() for p in front.col_emb.parameters())
    n_copy = sum(p.numel() for p in (front.copies.parameters() if front.copies is not None else []))
    log(f"trainable={n_tr/1e6:.3f}M (static={n_static/1e6:.3f}M, merger={n_merger/1e6:.3f}M, patch_pos={n_patch/1e6:.3f}M, copies={n_copy/1e6:.1f}M) vdim={vdim}")

    groups = [{"params": [p for p in params if all(p is not q for q in
                (list(front.static.parameters()) if front.static is not None else []))
                and all(p is not q for q in (list(front.copies.parameters()) if front.copies is not None else []))],
               "lr": args.lr}]
    if front.static is not None:
        groups.append({"params": list(front.static.parameters()), "lr": args.lr_static})
    if front.copies is not None:
        groups.append({"params": list(front.copies.parameters()), "lr": 1e-4})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    pre_hashes, pre_bb_sha1 = snapshot_backbone_hashes(model)
    verify_frozen_backbone_and_optimizer(model, opt, params)
    base_lrs = [g["lr"] for g in groups]

    def set_lr(step):
        f = min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f

    t0 = time.time()
    step, it = 0, iter(train_ld)
    hist = []
    while step < args.steps:
        step += 1
        set_lr(step)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                pv, grid, lens, tgts = next(it)
            except StopIteration:
                it = iter(train_ld)
                pv, grid, lens, tgts = next(it)
            v_hat, cl = front(pv.to(dev), grid, lens)
            loss, dcos = feature_loss(v_hat, tgts, cl)
            (loss / args.accum).backward()
            hist.append(dcos.item())
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            log(f"step {step} loss={loss.item():.4f} train_cos={sum(hist[-50:])/50:.4f} gn={gn.item():.1f} "
                f"{(time.time()-t0)/step:.2f}s/it")
        if step % args.eval_every == 0 or step == args.steps:
            c = evaluate(front, val_ld, dev)
            ex = {k: evaluate(front, ld, dev, 12) for k, ld in eval_lds.items()}
            log(f"  eval@{step}: val_cos={c:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in ex.items()))

    frozen_info = verify_frozen_backbone_and_optimizer(model, opt, params, pre_hashes)
    c = evaluate(front, val_ld, dev, 100)
    ex = {k: evaluate(front, ld, dev, 40) for k, ld in eval_lds.items()}
    log(f"FINAL val_cos={c:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in ex.items()))
    manifest = build_run_manifest("front_train", vars(args), split_info,
                                  extra={"frozen_check": frozen_info, "backbone_sha1_before": pre_bb_sha1})
    torch.save({"front": {k: v for k, v in front.state_dict().items()}, "args": vars(args), "vdim": vdim,
                "split_info": split_info, "manifest": manifest},
               args.out + ".pt")
    json.dump({"args": vars(args), "split_info": split_info, "manifest": manifest,
               "frozen_check": frozen_info, "val_cos": c, "extra_cos": ex,
               "trainable_M": n_tr / 1e6, "static_M": n_static / 1e6,
               "front_total_params": int(n_tr), "static_params": int(n_static),
               "merger_params": int(n_merger), "patch_pos_params": int(n_patch),
               "seconds": time.time() - t0,
               "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}, open(args.out + ".json", "w"), indent=1)
    log("FRONT_DONE")


if __name__ == "__main__":
    main()
