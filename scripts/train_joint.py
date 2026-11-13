"""叙事 1 + 2 合体（Stage 2B）：原图 → 前 K 层内置编码器 → v̂ → VRMG 动态旋转 → 纯文本通路。

推理时没有任何外部视觉塔，也没有图片 token 进入后 L-K 层。
--stage oracle        只评测：用老师视觉特征（叙事 1 的上限）
--stage front_frozen  只评测：用 Stage 2A 的前端产出 v̂，不联调
--stage joint         联调 front + VRMG（CE + KD + Rel + 特征对齐），结束时三种口径一并评测
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
import train_final as tf  # noqa: E402
from front import PatchFront, feature_loss  # noqa: E402
from provenance import capture_run_inputs, build_run_manifest, snapshot_backbone_hashes, verify_frozen_backbone_and_optimizer  # noqa: E402
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
STUDENT_LAYERS, TEACHER_LAYERS = tf.STUDENT_LAYERS, tf.TEACHER_LAYERS


def prepare_joint_split(data_spec, source_specs, seed=0):
    """联合训练/OPD 的数据加载与图像级切分入口（供训练与门禁脚本直接调用验证跨阶段一致性）。"""
    paths = {}
    for s in source_specs:
        paths.update(imgdata.parse_source(s)[1])
    data = tf.load_data(data_spec)
    train_raw, val_raw, split_info = tf.split_train_val_by_image(data, paths, seed=seed)
    train = [i for i in train_raw if data["ids"][i] in paths and data["vlen"][i] > 0]
    val = [i for i in val_raw if data["ids"][i] in paths and data["vlen"][i] > 0]
    return data, paths, train, val, split_info


@torch.no_grad()
def front_greedy_em(model, front, registry, tok, data, idx, paths, ip, dev, n=1000, max_new=24, bs=8):
    """无外部视觉塔模型统一验证集贪婪生成评测协议（max_new=24, bs=8, 左填充, stop_ids 截断）。"""
    model.eval(); front.eval()
    orig_side = tok.padding_side
    stops = tf._stop_ids(tok)
    stop_set = set(stops)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eval_idx = list(idx[:n])
    hit = stopped = tot_len = rep3_cnt = cnt = 0
    for s0 in range(0, len(eval_idx), bs):
        chunk = eval_idx[s0 : s0 + bs]
        pids = [tok(data["prompts"][i])["input_ids"] for i in chunk]
        pad = max(len(x) for x in pids)
        in_ids = torch.tensor([[pad_id] * (pad - len(x)) + x for x in pids], device=dev)
        attn = torch.tensor([[0] * (pad - len(x)) + [1] * len(x) for x in pids], device=dev)
        pvs, grids, lens = [], [], []
        for i in chunk:
            img = Image.open(paths[data["ids"][i]]).convert("RGB")
            o = ip(images=[img], return_tensors="pt")
            pvs.append(o["pixel_values"].to(torch.float16))
            grids.append(o["image_grid_thw"][0])
            lens.append(o["pixel_values"].shape[0])
        pv = torch.zeros(len(pvs), max(lens), pvs[0].shape[1], dtype=torch.float16, device=dev)
        for j, p in enumerate(pvs):
            pv[j, : lens[j]] = p.to(dev)
        v_hat, cl = front(pv, torch.stack(grids), lens)
        vis, vm = pad_vis([v_hat[j, : cl[j]].float() for j in range(len(cl))], dev)
        with rotation_context_with(registry, vis, vm, text_mask=attn):
            out = model.generate(input_ids=in_ids, attention_mask=attn, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=pad_id, eos_token_id=stops)
        for j, i in enumerate(chunk):
            gen = out[j, pad:]
            raw_ids = [int(x) for x in gen.tolist()]
            has_eos = any(t in stop_set for t in raw_ids)
            real = tf._cut_at_stop(gen, stop_set)
            pred = tok.decode(real, skip_special_tokens=True)
            hit += int(tf._norm(pred) == tf._norm(data["answers"][i]))
            stopped += int(has_eos)
            tot_len += len(real)
            ng3 = [tuple(real[m : m + 3]) for m in range(max(0, len(real) - 2))]
            rep3_cnt += int(len(ng3) >= 2 and len(set(ng3)) < len(ng3))
            cnt += 1
    tok.padding_side = orig_side
    front.train(); model.train()
    return {"em": hit / max(cnt, 1), "stop_rate": stopped / max(cnt, 1),
            "mean_len": tot_len / max(cnt, 1), "rep3_rate": rep3_cnt / max(cnt, 1), "em_n": cnt}


def make_collate(data, paths, ip, tok, seq, need_teacher):
    def fn(idx_list):
        ids, mask, labels, vis, vmask, k_ids, k_vals, k_mask, hidden = tf.collate(idx_list, data, tok, seq, need_teacher)
        pvs, grids, lens, tgts = [], [], [], []
        for j, i in enumerate(idx_list):
            img = Image.open(paths[data["ids"][i]]).convert("RGB")
            o = ip(images=[img], return_tensors="pt")
            pvs.append(o["pixel_values"].to(torch.float16))
            grids.append(o["image_grid_thw"][0])
            lens.append(o["pixel_values"].shape[0])
            tgts.append(vis[j, : data["vlen"][i]].clone())
        L = max(lens)
        pv = torch.zeros(len(idx_list), L, pvs[0].shape[1], dtype=torch.float16)
        for j, p in enumerate(pvs):
            pv[j, : lens[j]] = p
        return dict(ids=ids, mask=mask, labels=labels, k_ids=k_ids, k_vals=k_vals, k_mask=k_mask, hidden=hidden,
                    pv=pv, grid=torch.stack(grids), lens=lens, tgts=tgts,
                    a0s=[data["a_rel_start"][i] for i in idx_list], batch_idx=list(idx_list))
    return fn


def pad_vis(v_list, dev):
    n = max(v.shape[0] for v in v_list)
    out = torch.zeros(len(v_list), n, v_list[0].shape[-1], device=dev)
    m = torch.zeros(len(v_list), n, device=dev)
    for i, v in enumerate(v_list):
        out[i, : v.shape[0]] = v
        m[i, : v.shape[0]] = 1
    return out, m


@torch.no_grad()
def joint_eval(model, front, registry, tok, data, idx, paths, ip, dev, seq, source, n_acc=150, bs=8, shuf_seed=20261112):
    """source: 'front'（v̂）或 'teacher'（老师视觉特征）。返回 CE(配对/严格异图/无图) 与 teacher-forced 准确率。"""
    model.eval(); front.eval()
    idx = list(idx)
    shuf_idx, same_cnt, perm_sha1 = tf.build_index_derangement(idx, data, paths, seed=shuf_seed)
    assert same_cnt == 0, f"joint_eval 异图置换存在 {same_cnt} 对同图样本！"
    pos_of = {orig_i: p for p, orig_i in enumerate(idx)}
    perm_pos = [pos_of[s_i] for s_i in shuf_idx]

    dl = DataLoader(idx, batch_size=bs, collate_fn=make_collate(data, paths, ip, tok, seq, False),
                    num_workers=2, shuffle=False)
    batches_meta = []
    vis_by_pos = []
    for b in dl:
        if source == "front":
            v_hat, cl = front(b["pv"].to(dev), b["grid"], b["lens"])
            vis_list = [v_hat[i, : cl[i]].float().cpu() for i in range(len(cl))]
        else:
            vis_list = [t.float().cpu() for t in b["tgts"]]
        vis_by_pos.extend(vis_list)
        batches_meta.append((b["ids"], b["mask"], b["labels"], b["a0s"], len(vis_list)))

    tot = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    cnt = 0
    offset = 0
    for ids_cpu, mask_cpu, labels_cpu, a0s, b_len in batches_meta:
        ids, mask, labels = ids_cpu.to(dev), mask_cpu.to(dev), labels_cpu.to(dev)
        vis_real_list = [vis_by_pos[offset + k].to(dev) for k in range(b_len)]
        vis_shuf_list = [vis_by_pos[perm_pos[offset + k]].to(dev) for k in range(b_len)]
        vis, vm = pad_vis(vis_real_list, dev)
        vis_shuf, vm_shuf = pad_vis(vis_shuf_list, dev)
        for key, v, m in (("correct", vis, vm), ("shuffled", vis_shuf, vm_shuf), ("no_image", None, None)):
            if v is None:
                lg = model(input_ids=ids, attention_mask=mask).logits.float()
            else:
                with rotation_context_with(registry, v, m, ans_start=a0s, text_mask=mask):
                    lg = model(input_ids=ids, attention_mask=mask).logits.float()
            tot[key] += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1)).item()
        cnt += 1
        offset += b_len
    ce = {k: v / max(cnt, 1) for k, v in tot.items()}

    first = full = n = 0
    for pos, i in enumerate(idx[:n_acc]):
        text = data["prompts"][i] + " " + data["answers"][i]
        tid = tok(text)["input_ids"]
        a0, na = data["a_rel_start"][i], data["n_answer"][i]
        vis = vis_by_pos[pos].unsqueeze(0).to(dev).float()
        vm = torch.ones(1, vis.shape[1], device=dev)
        with rotation_context_with(registry, vis, vm, ans_start=[a0], text_mask=torch.ones(1, len(tid), device=dev)):
            lg = model(input_ids=torch.tensor([tid], device=dev)).logits[0].float()
        preds = [lg[a0 - 1 + k].argmax().item() for k in range(na)]
        first += int(preds[0] == tid[a0]); full += int(preds == tid[a0 : a0 + na]); n += 1
    front.train()
    return {**ce, "gap": ce["shuffled"] - ce["correct"], "tf_first": first / max(n, 1), "tf_full": full / max(n, 1),
            "shuf_same_image_count": same_cnt, "shuf_perm_sha1": perm_sha1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["oracle", "front_frozen", "joint"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", action="append", required=True, help="meta|manifest|img_root（用于解析 id→图片路径）")
    ap.add_argument("--vrmg-ckpt", required=True)
    ap.add_argument("--front-ckpt", default="")
    ap.add_argument("--front-mode", default="rot")
    ap.add_argument("--multifreq", action="store_true")
    ap.add_argument("--tokrot", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--prompt-alpha", type=float, default=1.0, help="Prompt 段旋转强度掩码（<1 保留原 LLM 读题能力）")
    ap.add_argument("--hyb-bound", type=float, default=0.0, help="hybrid 残差相对范数上限（>0 防长序列激活爆炸）")
    ap.add_argument("--qcond", action="store_true", help="问题条件化旋转：把题目文本池化向量叠到 VRMG queries")
    ap.add_argument("--direct-hyb", action="store_true", help="hybrid 直接在 256 维对全图视觉 cell 做无损交叉注意力")
    ap.add_argument("--no-rot", action="store_true", help="旋转恒等映射 R=I，仅保留视觉 cross-attention")
    ap.add_argument("--rot-bound", type=float, default=0.0, help="李代数切空间有界正交旋转半径（>0 启用）")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--teacher", default=imgdata.TEACHER)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积：每个优化步用 accum 个 bs 大小的微批")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr-vrmg", type=float, default=2e-4)
    ap.add_argument("--lr-front", type=float, default=1e-4)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--rel-weight", type=float, default=0.1)
    ap.add_argument("--feat-weight", type=float, default=1.0)
    ap.add_argument("--gnd-weight", type=float, default=0.0, help="带梯度的同 batch 错图对比损失权重（>0 启用）")
    ap.add_argument("--gnd-margin", type=float, default=0.75, help="首内容 token 错图与真图 NLL 边际阈值")
    ap.add_argument("--arm", choices=["ce", "kd", "rel"], default="rel")
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--n-em", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student-layers", default="6,13,20,27")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    capture_run_inputs(vars(args))
    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(args.out + ".log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    data, paths, train, val, split_info = prepare_joint_split(args.data, args.source, seed=args.seed)
    val_eval = val[: args.n_eval]
    val_em_1000, val_1000_sha1 = tf.select_stratified_val_1000(val, data, n_target=args.n_em, seed=0)
    assert train and val, "按图切分后可用 train/val 为空"

    ip = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True).image_processor
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    ck = torch.load(args.vrmg_ckpt, map_location=dev)
    a1 = ck["args"]
    layers = [int(x) for x in a1["layers"].split(",") if x]
    K = 0 if args.front_mode == "none" else args.K
    pa = args.prompt_alpha if args.prompt_alpha != 1.0 else a1.get("prompt_alpha", 1.0)
    hb = args.hyb_bound if args.hyb_bound > 0.0 else a1.get("hyb_bound", 0.0)
    rb = args.rot_bound if args.rot_bound > 0.0 else float(a1.get("rot_bound", 0.0))
    args.prompt_alpha, args.hyb_bound, args.rot_bound = pa, hb, rb
    vrmgs, registry = attach_rotation(model, layers, block_size=a1["block"], multifreq=args.multifreq or a1.get("multifreq", False), tokrot=args.tokrot or a1.get("tokrot", False), hybrid=args.hybrid or a1.get("hybrid", False),
                                      visual_dim=data["vfeat"][0].shape[-1], query_dim=a1["query_dim"],
                                      wrap_extra=[l for l in range(K) if l not in layers],
                                      prompt_alpha=pa, hyb_bound=hb,
                                      qcond=args.qcond or a1.get("qcond", False),
                                      direct_hyb=args.direct_hyb or a1.get("direct_hyb", False),
                                      no_rot=args.no_rot or (a1.get("no_rot", False) and rb == 0.0 and not args.multifreq),
                                      rot_bound=rb)
    for m, sd in zip(vrmgs, ck["state"]):
        m.load_state_dict(sd, strict=False)
    front = PatchFront(model, K=K, mode=args.front_mode, vdim=data["vfeat"][0].shape[-1],
                       lora_rank=args.lora_rank).to(dev)
    if args.front_ckpt:
        front.load_state_dict(torch.load(args.front_ckpt, map_location=dev)["front"])
    log(f"=== joint stage={args.stage} front={args.front_mode} K={K} arm={args.arm} train={len(train)} "
        f"val={len(val)} eval_n={len(val_eval)} split={split_info} ===")
    n_front_tot = sum(p.numel() for p in front.trainable_params())
    n_static = sum(p.numel() for p in (front.static.parameters() if front.static is not None else []))
    n_merger = sum(p.numel() for p in front.merger.parameters())
    n_vrmg = sum(p.numel() for m in vrmgs for p in m.parameters())
    result = {"args": vars(args), "split_info": split_info,
              "param_counts": {"front_total": int(n_front_tot), "static_only": int(n_static),
                               "merger_only": int(n_merger), "vrmg_total": int(n_vrmg),
                               "joint_trainable_total": int(n_front_tot + n_vrmg)}}
    t0 = time.time()

    pre_hashes, pre_bb_sha1 = snapshot_backbone_hashes(model)
    opt = None
    all_params = []
    if args.stage == "joint":
        fparams = front.trainable_params()
        vparams = [p for m in vrmgs for p in m.parameters()]
        freeze_all_but(model, [])
        for p in fparams + vparams:
            p.requires_grad_(True)
        rel_head = None
        if args.arm == "rel":
            rel_head = nn.Linear(model.config.hidden_size, data["hidden"][0][0].shape[-1], bias=False).to(dev)
            if ck.get("rel_head"):
                rel_head.load_state_dict(ck["rel_head"])
        groups = []
        if args.rot_bound > 0.0:
            ca_p, rot_p = [], []
            for m in vrmgs:
                for n_p, p in m.named_parameters():
                    (ca_p if n_p.startswith(("anc_", "hyb_")) else rot_p).append(p)
            if ca_p:
                groups.append({"params": ca_p, "lr": args.lr_vrmg, "weight_decay": 0.01})
            if rot_p:
                groups.append({"params": rot_p, "lr": args.lr_vrmg * 0.25, "weight_decay": 0.01})
        else:
            groups.append({"params": vparams, "lr": args.lr_vrmg, "weight_decay": 0.01})

        static_all = list(front.static.parameters()) if front.static is not None else []
        static_ids = {id(p) for p in static_all}
        f_base = [p for p in fparams if id(p) not in static_ids]
        groups.append({"params": f_base, "lr": args.lr_front, "weight_decay": 0.01})
        if front.static is not None:
            scale_mod = getattr(front.static, "scale", None)
            scale_ps = list(scale_mod.parameters()) if isinstance(scale_mod, nn.Module) else []
            scale_ids = {id(p) for p in scale_ps}
            s_rot_ps = [p for p in static_all if id(p) not in scale_ids]
            lr_st = max(args.lr_front * 2.5, 2.5e-4)
            if s_rot_ps:
                groups.append({"params": s_rot_ps, "lr": lr_st, "weight_decay": 0.01})
            if scale_ps:
                groups.append({"params": scale_ps, "lr": lr_st, "weight_decay": 0.0})
        if rel_head is not None:
            groups.append({"params": list(rel_head.parameters()), "lr": args.lr_vrmg, "weight_decay": 0.01})
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        all_params = fparams + vparams + (list(rel_head.parameters()) if rel_head is not None else [])
        verify_frozen_backbone_and_optimizer(model, opt, all_params)
        base_lrs = [g["lr"] for g in groups]
        dl = DataLoader(train, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                        collate_fn=make_collate(data, paths, ip, tok, args.seq, args.arm != "ce"),
                        persistent_workers=True)
        assert len(dl) > 0, "训练数据不足一个 drop_last batch"
        it, hist = iter(dl), []
        all_params = fparams + vparams + (list(rel_head.parameters()) if rel_head is not None else [])
        for ms in range(1, args.steps * args.accum + 1):
            step, micro = (ms - 1) // args.accum + 1, (ms - 1) % args.accum
            try:
                b = next(it)
            except StopIteration:
                it = iter(dl)
                b = next(it)
            if micro == 0:
                f = min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * min(step, args.steps) / args.steps))
                for g, bl in zip(opt.param_groups, base_lrs):
                    g["lr"] = bl * f
                opt.zero_grad(set_to_none=True)
            ids, mask, labels = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            v_hat, cl = front(b["pv"].to(dev), b["grid"], b["lens"])
            fl, _ = feature_loss(v_hat, b["tgts"], cl)
            vis, vm = pad_vis([v_hat[i, : cl[i]] for i in range(len(cl))], dev)
            with rotation_context_with(registry, vis, vm, ans_start=b["a0s"], text_mask=mask):
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=(args.arm == "rel"))
            lg = out.logits.float()[:, :-1]
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1))
            loss = ce + args.feat_weight * fl
            if args.gnd_weight > 0.0 and ids.shape[0] >= 2:
                perm_list, valid_list = tf.build_inbatch_diff_image_perm(b["batch_idx"], data, paths)
                if any(v > 0 for v in valid_list):
                    perm = torch.tensor(perm_list, device=dev)
                    valid_m = torch.tensor(valid_list, device=dev)
                    with rotation_context_with(registry, vis[perm], vm[perm], ans_start=b["a0s"], text_mask=mask):
                        out_shuf = model(input_ids=ids, attention_mask=mask)
                    lg_shuf = out_shuf.logits.float()[:, :-1]
                    p0_list = []
                    for j, a0 in enumerate(b["a0s"]):
                        if a0 + 1 < ids.shape[1] and int(ids[j, a0].item()) == 220:
                            a0 = a0 + 1
                        p0_list.append(min(max(0, a0 - 1), lg.shape[1] - 1))
                    p0 = torch.tensor(p0_list, device=dev)
                    b_idx = torch.arange(ids.shape[0], device=dev)
                    y0 = ids[b_idx, p0 + 1]
                    nll_real = F.cross_entropy(lg[b_idx, p0], y0, reduction="none")
                    nll_shuf = F.cross_entropy(lg_shuf[b_idx, p0], y0, reduction="none")
                    gnd_raw = F.relu(args.gnd_margin - (nll_shuf - nll_real))
                    gnd = (gnd_raw * valid_m).sum() / valid_m.sum().clamp(min=1.0)
                    loss = loss + args.gnd_weight * gnd
            if args.arm in ("kd", "rel"):
                k_ids, k_vals, k_mask = b["k_ids"].to(dev), b["k_vals"].to(dev), b["k_mask"].to(dev)
                kd = tf.topk_conditional_kl(lg, k_ids, k_vals, k_mask, args.kd_temp)
                loss = loss + args.kd_weight * kd
                if args.arm == "rel":
                    terms = []
                    for li, sl in enumerate([int(x) for x in args.student_layers.split(',')]):
                        hs = F.normalize(rel_head(out.hidden_states[sl][:, :-1].float()), dim=-1)
                        th = F.normalize(b["hidden"][li].to(dev).float(), dim=-1)
                        terms.append(((1.0 - (hs * th).sum(-1)) * k_mask).sum() / k_mask.sum().clamp(min=1.0))
                    loss = loss + args.rel_weight * torch.stack(terms).mean()
            (loss / args.accum).backward()
            hist.append((ce.item(), fl.item()))
            if micro < args.accum - 1:
                continue
            gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()
            if step % 50 == 0:
                h = hist[-50:]
                log(f"step {step} ce={sum(x[0] for x in h)/50:.4f} feat={sum(x[1] for x in h)/50:.4f} "
                    f"gn={gn.item():.1f} {(time.time()-t0)/step:.2f}s/it")
            if step % max(args.steps // 4, 200) == 0 and step < args.steps:
                ev = joint_eval(model, front, registry, tok, data, val_eval[:160], paths, ip, dev, args.seq, "front",
                                n_acc=60)
                log(f"  eval@{step} (front v̂): " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, (int, float))))
        result["train_seconds"] = time.time() - t0

    frozen_info = verify_frozen_backbone_and_optimizer(model, opt, all_params if all_params else None, pre_hashes)
    srcs = {"oracle": ["teacher"], "front_frozen": ["front"], "joint": ["front", "teacher"]}[args.stage]
    result["eval"] = {}
    for s in srcs:
        ev = joint_eval(model, front, registry, tok, data, val_eval, paths, ip, dev, args.seq, s)
        if s == "front":
            ev.update(front_greedy_em(model, front, registry, tok, data, val_em_1000, paths, ip, dev,
                                      n=len(val_em_1000), max_new=24, bs=8))
            ev["val_em_sha1"] = val_1000_sha1
        result["eval"][s] = ev
        log(f"FINAL[{s}] " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, (int, float))))
    manifest = build_run_manifest("joint_train", vars(args), split_info,
                                  extra={"frozen_check": frozen_info, "val_em_sha1": val_1000_sha1,
                                         "backbone_sha1_before": pre_bb_sha1})
    result["manifest"] = manifest
    result["frozen_check"] = frozen_info
    result["peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    json.dump(result, open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
    if args.stage == "joint":
        torch.save({"front": front.state_dict(), "vrmg": [m.state_dict() for m in vrmgs],
                    "args": vars(args), "manifest": manifest},
                   args.out + ".pt")
    log("JOINT_DONE")


if __name__ == "__main__":
    main()
