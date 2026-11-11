"""SpinVL-Tok：无塔旋转模型（PatchFront v̂ → VRMG 旋转）+ 少量视觉 token 拼进文本序列。

动机（2026-11-06 实测定位）：纯旋转只把图像压成一组全局旋转参数，读不了图里的字符、坐标轴数值与几何标注，
所以 6 榜里的开放题（ChartQA 0.24%、OCRBench 0.00%、Geometry3K 1.83%）远低于 LLaVA 式拼接基线（10.40 / 1.60 / 6.49）。
这里把 v̂ 自适应池化成 n_vis 个 token（32/64，远少于 LLaVA 的 576）拼在文本前，既恢复细粒度读取，又保留大部分 FLOPs 优势。
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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
import train_final as tf  # noqa: E402
from front import PatchFront, feature_loss  # noqa: E402
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"


def make_collate(data, paths, ip, tok, seq):
    def fn(idx_list):
        ids, mask, labels, vis, vmask, *_ = tf.collate(idx_list, data, tok, seq, False)
        pvs, grids, lens, tgts = [], [], [], []
        for j, i in enumerate(idx_list):
            o = ip(images=[Image.open(paths[data["ids"][i]]).convert("RGB")], return_tensors="pt")
            pvs.append(o["pixel_values"].to(torch.float16))
            grids.append(o["image_grid_thw"][0])
            lens.append(o["pixel_values"].shape[0])
            tgts.append(vis[j, : data["vlen"][i]].clone())
        L = max(lens)
        pv = torch.zeros(len(idx_list), L, pvs[0].shape[1], dtype=torch.float16)
        for j, p in enumerate(pvs):
            pv[j, : lens[j]] = p
        return dict(ids=ids, mask=mask, labels=labels, pv=pv, grid=torch.stack(grids), lens=lens, tgts=tgts)
    return fn


def pad_vis(v_list, dev):
    n = max(v.shape[0] for v in v_list)
    out = torch.zeros(len(v_list), n, v_list[0].shape[-1], device=dev)
    m = torch.zeros(len(v_list), n, device=dev)
    for i, v in enumerate(v_list):
        out[i, : v.shape[0]] = v
        m[i, : v.shape[0]] = 1
    return out, m


def prepend(model, vt, ids, mask, labels=None):
    """把 n_vis 个视觉 token 拼在文本 embedding 前面，同步扩展 attention mask 与 labels。"""
    emb = model.get_input_embeddings()(ids)
    emb = torch.cat([vt.to(emb.dtype), emb], dim=1)
    am = torch.cat([torch.ones(ids.shape[0], vt.shape[1], device=ids.device, dtype=mask.dtype), mask], dim=1)
    if labels is None:
        return emb, am
    lab = torch.cat([torch.full((ids.shape[0], vt.shape[1]), -100, device=ids.device, dtype=labels.dtype), labels], dim=1)
    return emb, am, lab


@torch.no_grad()
def joint_eval(model, front, registry, tok, data, idx, paths, ip, dev, seq, n_acc=150, bs=8):
    model.eval(); front.eval()
    nv = front.n_vis
    tot = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    cnt = 0
    dl = DataLoader(idx, batch_size=bs, collate_fn=make_collate(data, paths, ip, tok, seq), num_workers=2)
    for b in dl:
        ids, mask, labels = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
        v_hat, cl = front(b["pv"].to(dev), b["grid"], b["lens"])
        vt = front.tokens(v_hat, cl)
        perm = torch.tensor([(i + 1) % len(vt) for i in range(len(vt))])
        for key, v in (("correct", vt), ("shuffled", vt[perm]), ("no_image", None)):
            if v is None:
                lg = model(input_ids=ids, attention_mask=mask).logits.float()
                lab = labels
            else:
                emb, am, lab = prepend(model, v, ids, mask, labels)
                with rotation_context_with(registry, *pad_vis([v_hat[i, : cl[i]].float() for i in range(len(cl))], dev)):
                    lg = model(inputs_embeds=emb, attention_mask=am).logits.float()
            tot[key] += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1),
                                        ignore_index=-100).item()
        cnt += 1
    ce = {k: v / max(cnt, 1) for k, v in tot.items()}

    first = full = n = 0
    for i in idx[:n_acc]:
        tid = tok(data["prompts"][i] + " " + data["answers"][i])["input_ids"]
        a0, na = data["a_rel_start"][i], data["n_answer"][i]
        o = ip(images=[Image.open(paths[data["ids"][i]]).convert("RGB")], return_tensors="pt")
        v_hat, cl = front(o["pixel_values"].to(torch.float16).unsqueeze(0).to(dev), o["image_grid_thw"],
                          [o["pixel_values"].shape[0]])
        vt = front.tokens(v_hat, cl)
        ids = torch.tensor([tid], device=dev)
        emb, am = prepend(model, vt, ids, torch.ones(1, len(tid), device=dev, dtype=torch.long))
        with rotation_context_with(registry, *pad_vis([v_hat[0, : cl[0]].float()], dev)):
            lg = model(inputs_embeds=emb, attention_mask=am).logits[0].float()
        preds = [lg[nv + a0 - 1 + k].argmax().item() for k in range(na)]
        first += int(preds[0] == tid[a0]); full += int(preds == tid[a0 : a0 + na]); n += 1
    front.train()
    return {**ce, "gap": ce["shuffled"] - ce["correct"], "tf_first": first / max(n, 1), "tf_full": full / max(n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--vrmg-ckpt", required=True)
    ap.add_argument("--front-ckpt", default="")
    ap.add_argument("--front-mode", default="rots")
    ap.add_argument("--multifreq", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--n-vis", type=int, default=32)
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr-vrmg", type=float, default=2e-4)
    ap.add_argument("--lr-front", type=float, default=1e-4)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(args.out + ".log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    paths = {}
    for s in args.source:
        paths.update(imgdata.parse_source(s)[1])
    data = tf.load_data(args.data)
    train, val, _ = tf.split_train_val_by_image(data, paths, seed=args.seed)
    val_eval = val[: args.n_eval]

    ip = AutoProcessor.from_pretrained(imgdata.TEACHER, trust_remote_code=True).image_processor
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    ck = torch.load(args.vrmg_ckpt, map_location=dev)
    a1 = ck["args"]
    layers = [int(x) for x in a1["layers"].split(",") if x]
    K = 0 if args.front_mode == "none" else args.K
    vrmgs, registry = attach_rotation(model, layers, block_size=a1["block"],
                                      multifreq=args.multifreq or a1.get("multifreq", False),
                                      hybrid=args.hybrid or a1.get("hybrid", False),
                                      visual_dim=data["vfeat"][0].shape[-1], query_dim=a1["query_dim"],
                                      wrap_extra=[l for l in range(K) if l not in layers])
    for m, sd in zip(vrmgs, ck["state"]):
        m.load_state_dict(sd)
    front = PatchFront(model, K=K, mode=args.front_mode, vdim=data["vfeat"][0].shape[-1],
                       lora_rank=args.lora_rank, n_vis=args.n_vis).to(dev)
    if args.front_ckpt:
        front.load_state_dict(torch.load(args.front_ckpt, map_location=dev)["front"], strict=False)
    log(f"=== SpinVL-Tok joint front={args.front_mode} K={K} n_vis={args.n_vis} train={len(train)} val={len(val)} ===")
    result = {"args": vars(args)}
    t0 = time.time()

    fparams = front.trainable_params()
    vparams = [p for m in vrmgs for p in m.parameters()]
    freeze_all_but(model, [])
    for p in fparams + vparams:
        p.requires_grad_(True)
    groups = [{"params": vparams, "lr": args.lr_vrmg}, {"params": fparams, "lr": args.lr_front}]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base_lrs = [g["lr"] for g in groups]
    dl = DataLoader(train, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                    collate_fn=make_collate(data, paths, ip, tok, args.seq), persistent_workers=args.workers > 0)
    it, step, hist, micro = iter(dl), 0, [], 0
    all_params = fparams + vparams
    while step < args.steps:
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); continue
        if micro == 0:
            step += 1
            f = min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * f
            opt.zero_grad(set_to_none=True)
        ids, mask, labels = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
        v_hat, cl = front(b["pv"].to(dev), b["grid"], b["lens"])
        fl, _ = feature_loss(v_hat, b["tgts"], cl)
        vt = front.tokens(v_hat, cl)
        emb, am, lab = prepend(model, vt, ids, mask, labels)
        with rotation_context_with(registry, *pad_vis([v_hat[i, : cl[i]] for i in range(len(cl))], dev)):
            lg = model(inputs_embeds=emb, attention_mask=am).logits.float()[:, :-1]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
        loss = ce + args.feat_weight * fl
        (loss / args.accum).backward()
        hist.append((ce.item(), fl.item()))
        micro += 1
        if micro < args.accum:
            continue
        micro = 0
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        if step % 50 == 0:
            h = hist[-50 * args.accum:]
            log(f"step {step} ce={sum(x[0] for x in h)/len(h):.4f} feat={sum(x[1] for x in h)/len(h):.4f} "
                f"gn={gn.item():.1f} {(time.time()-t0)/step:.2f}s/it")
        if step % max(args.steps // 4, 200) == 0 and step < args.steps:
            ev = joint_eval(model, front, registry, tok, data, val_eval[:160], paths, ip, dev, args.seq, n_acc=60)
            log(f"  eval@{step}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
    result["train_seconds"] = time.time() - t0
    ev = joint_eval(model, front, registry, tok, data, val_eval, paths, ip, dev, args.seq)
    result["eval"] = {"tok": ev}
    log("FINAL[tok] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
    result["peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    json.dump(result, open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
    torch.save({"front": front.state_dict(), "vrmg": [m.state_dict() for m in vrmgs], "args": vars(args)},
               args.out + ".pt")
    log("JOINT_TOK_DONE")


if __name__ == "__main__":
    main()
