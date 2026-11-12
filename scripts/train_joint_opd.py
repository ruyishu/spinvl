"""无外部视觉塔模型的 VL-OPD 在线策略蒸馏（Stage 2C：Tower-Free VL-OPD）。

核心叙事（1 → 2 串联）：
1. 第一步已经用「前 4 层内置读图模块 PatchFront + 后 24 层动态旋转 VRMG」彻底干掉了外部视觉塔（SigLIP2 / Qwen-ViT），
   得到一个直接吃原图 patch 的纯文模型起点（--joint-ckpt）。
2. 因为它骨子里仍是纯文本模型、缺少数亿张图的视觉塔预训练，直接看图生成回答时理解不够准；
   本脚本以该无塔模型为学生起点，让学生直接看原图 rollout 生成回答 ŷ，再由带原装视觉塔的
   Qwen3-VL-8B 老师看同一张原图，在学生的轨迹 ŷ 上做逐词在线 KL 纠偏，梯度同时更新学生的
   前 4 层内置看图模块（PatchFront）与后 24 层动态旋转（VRMG）。
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
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
import train_final as tf  # noqa: E402
import train_joint as tj  # noqa: E402
import train_opd as topd  # noqa: E402
from front import PatchFront, feature_loss  # noqa: E402
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"


@torch.no_grad()
def front_greedy_em(model, front, registry, tok, data, idx, paths, ip, dev, n=1000, max_new=24, bs=8):
    """无外部视觉塔模型自己看原图 → PatchFront → VRMG 批量贪婪生成整段答案，算 EM / 停止率 / 长度 / 3-gram 复读率。"""
    model.eval(); front.eval()
    stops = topd.stop_ids(tok)
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
        vis, vm = tj.pad_vis([v_hat[j, : cl[j]].float() for j in range(len(cl))], dev)
        with rotation_context_with(registry, vis, vm, text_mask=attn):
            out = model.generate(input_ids=in_ids, attention_mask=attn, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=pad_id, eos_token_id=stops)
        for j, i in enumerate(chunk):
            gen = out[j, pad:]
            raw_ids = [int(x) for x in gen.tolist()]
            has_eos = any(t in stop_set for t in raw_ids)
            real = topd.cut_at_stop(gen, stop_set)
            pred = tok.decode(real, skip_special_tokens=True)
            hit += int(tf._norm(pred) == tf._norm(data["answers"][i]))
            stopped += int(has_eos)
            tot_len += len(real)
            ng3 = [tuple(real[m : m + 3]) for m in range(max(0, len(real) - 2))]
            rep3_cnt += int(len(ng3) >= 2 and len(set(ng3)) < len(ng3))
            cnt += 1
    front.train(); model.train()
    return {"em": hit / max(cnt, 1), "stop_rate": stopped / max(cnt, 1),
            "mean_len": tot_len / max(cnt, 1), "rep3_rate": rep3_cnt / max(cnt, 1), "em_n": cnt}


def full_front_eval(model, front, registry, tok, data, val_loss_idx, val_em_idx, paths, ip, dev, seq, n_acc=150, n_em=1000, max_new=24):
    ev = tj.joint_eval(model, front, registry, tok, data, val_loss_idx, paths, ip, dev, seq, "front", n_acc=n_acc)
    ev.update(front_greedy_em(model, front, registry, tok, data, val_em_idx, paths, ip, dev, n=n_em, max_new=max_new))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--joint-ckpt", required=True, help="第一步训好的无塔模型权重（含 front 与 vrmg）")
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--teacher", default=imgdata.TEACHER)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积：每个优化步用 accum 个 bs 大小的微批")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--min-new", type=int, default=1)
    ap.add_argument("--gen-temp", type=float, default=0.7)
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--kl-mode", choices=["none", "offkd", "fwd", "rev", "jsd", "seqkd"], default="fwd",
                    help="后训练类型：none=C0(仅CE+feat), offkd=C1(金标前缀离线KD), fwd=C2(Forward-KL OPD), jsd=C3(JSD OPD)")
    ap.add_argument("--kl-weight", type=float, default=0.25, help="蒸馏损失权重（C0 设为 0，C1/C2/C3 默认 0.25）")
    ap.add_argument("--ce-weight", type=float, default=1.0)
    ap.add_argument("--feat-weight", type=float, default=0.2)
    ap.add_argument("--prompt-alpha", type=float, default=1.0, help="Prompt 段旋转强度掩码（<1 保留原 LLM 读题能力）")
    ap.add_argument("--hyb-bound", type=float, default=0.0, help="hybrid 残差相对范数上限（>0 防长序列激活爆炸）")
    ap.add_argument("--multifreq", action="store_true")
    ap.add_argument("--tokrot", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--qcond", action="store_true", help="问题条件化旋转：把题目文本池化向量叠到 VRMG queries")
    ap.add_argument("--direct-hyb", action="store_true", help="hybrid 直接在 256 维对全图视觉 cell 做无损交叉注意力")
    ap.add_argument("--no-rot", action="store_true", help="旋转恒等映射 R=I，仅保留视觉 cross-attention")
    ap.add_argument("--lr-vrmg", type=float, default=2e-5)
    ap.add_argument("--lr-front", type=float, default=1e-5)
    ap.add_argument("--tune-backbone", action="store_true", help="OPD 同时更新学生 LLM 主干")
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--n-acc", type=int, default=150)
    ap.add_argument("--n-em", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
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
    n = len(data["ids"])
    train, val, split_info = tf.split_train_val_by_image(data, paths, seed=args.seed)
    val = [i for i in val if data["ids"][i] in paths]
    train = [i for i in train if data["ids"][i] in paths]
    assert train and val, "按图切分后可用 train/val 为空"
    val_eval = val[: args.n_eval]
    val_em_1000, val_1000_sha1 = tf.select_stratified_val_1000(val, data, n_target=args.n_em, seed=0)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stops = topd.stop_ids(tok)
    pad_id = tok.pad_token_id

    proc = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    ip = proc.image_processor

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    jck = torch.load(args.joint_ckpt, map_location=dev)
    ja = jck["args"]
    vck = torch.load(ja["vrmg_ckpt"], map_location=dev)
    a1 = vck["args"]
    layers = [int(x) for x in a1["layers"].split(",") if x]
    K = 0 if ja.get("front_mode") == "none" else int(ja.get("K", 4))
    multifreq = bool(ja.get("multifreq", False) or a1.get("multifreq", False))
    tokrot = bool(ja.get("tokrot", False) or a1.get("tokrot", False))
    hybrid = bool(ja.get("hybrid", False) or a1.get("hybrid", False))

    pa = args.prompt_alpha if args.prompt_alpha != 1.0 else ja.get("prompt_alpha", a1.get("prompt_alpha", 1.0))
    hb = args.hyb_bound if args.hyb_bound > 0.0 else ja.get("hyb_bound", a1.get("hyb_bound", 0.0))
    ja["prompt_alpha"], ja["hyb_bound"] = pa, hb
    vrmgs, registry = attach_rotation(
        model, layers, block_size=a1["block"], visual_dim=data["vfeat"][0].shape[-1],
        query_dim=a1["query_dim"], wrap_extra=[l for l in range(K) if l not in layers],
        multifreq=multifreq, tokrot=tokrot, hybrid=hybrid, prompt_alpha=pa, hyb_bound=hb,
        qcond=bool(args.qcond or ja.get("qcond", False) or a1.get("qcond", False)),
        direct_hyb=bool(args.direct_hyb or ja.get("direct_hyb", False) or a1.get("direct_hyb", False)),
        no_rot=bool(args.no_rot or ja.get("no_rot", False) or a1.get("no_rot", False)),
    )
    for m, sd in zip(vrmgs, jck["vrmg"]):
        m.load_state_dict(sd)
    front = PatchFront(model, K=K, mode=ja.get("front_mode", "rots"),
                       vdim=data["vfeat"][0].shape[-1], lora_rank=ja.get("lora_rank", 16)).to(dev)
    front.load_state_dict(jck["front"])

    fparams = front.trainable_params()
    vparams = [p for m in vrmgs for p in m.parameters()]
    freeze_all_but(model, [])
    for p in fparams + vparams:
        p.requires_grad_(True)
    bparams = []
    if args.tune_backbone:
        bparams = list(model.parameters())
        for p in bparams:
            p.requires_grad_(True)
    backbone_params = [p for p in bparams if all(p is not q for q in fparams + vparams)]

    need_teacher = (args.kl_mode != "none" and args.kl_weight > 0.0)
    teacher = None
    if need_teacher:
        teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map=dev,
                                                              trust_remote_code=True).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    log(f"=== Tower-Free Post-Train mode={args.kl_mode} kl_w={args.kl_weight} ce_w={args.ce_weight} feat_w={args.feat_weight} "
        f"ckpt={os.path.basename(args.joint_ckpt)} front={ja.get('front_mode')} K={K} "
        f"mf={multifreq} hybrid={hybrid} steps={args.steps} bs={args.bs} val_em_n={len(val_em_1000)} ===")

    result = {"args": vars(args), "joint_args": ja, "split_info": split_info, "val_em_sha1": val_1000_sha1}
    init_ev = full_front_eval(model, front, registry, tok, data, val_eval, val_em_1000, paths, ip, dev, args.seq,
                              n_acc=args.n_acc, n_em=len(val_em_1000), max_new=args.max_new)
    result["init_eval"] = init_ev
    log("INIT[front v̂]: " + " ".join(f"{k}={v:.4f}" for k, v in init_ev.items() if isinstance(v, (int, float))))

    opt = torch.optim.AdamW([{"params": vparams, "lr": args.lr_vrmg},
                             {"params": fparams, "lr": args.lr_front}] +
                            ([{"params": backbone_params, "lr": args.backbone_lr}] if backbone_params else []),
                            weight_decay=0.01)
    if not args.tune_backbone:
        opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        allowed_ids = {id(p) for p in fparams + vparams}
        assert opt_ids == allowed_ids, "Optimizer 包含非白名单参数！"
        assert all(not p.requires_grad for p in model.parameters()), "冻结模式下原始 LLM 存在 requires_grad=True 参数！"
    base_lrs = [g["lr"] for g in opt.param_groups]
    all_params = fparams + vparams + backbone_params
    collate_fn = tj.make_collate(data, paths, ip, tok, args.seq, False)
    t0, hist = time.time(), []
    eos_id = tok.eos_token_id

    for ms in range(1, args.steps * args.accum + 1):
        step, micro = (ms - 1) // args.accum + 1, (ms - 1) % args.accum
        if micro == 0:
            f = min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            for g, bl in zip(opt.param_groups, base_lrs):
                g["lr"] = bl * f
            opt.zero_grad(set_to_none=True)
        idx = [train[(ms * args.bs + k) % len(train)] for k in range(args.bs)]
        b = collate_fn(idx)
        prompts = [data["prompts"][i] for i in idx]
        pids = [tok(p)["input_ids"] for p in prompts]
        pad = max(len(x) for x in pids)
        in_ids = torch.tensor([[pad_id] * (pad - len(x)) + x for x in pids], device=dev)
        attn = torch.tensor([[0] * (pad - len(x)) + [1] * len(x) for x in pids], device=dev)

        kl_terms = []
        gids = [[] for _ in idx]
        if need_teacher:
            if args.kl_mode == "offkd":
                # C1：使用金标答案（含 EOS）作为对齐前缀轨迹
                gids = [tok(" " + data["answers"][i], add_special_tokens=False)["input_ids"][: args.max_new - 1]
                        + ([eos_id] if eos_id is not None else []) for i in idx]
            else:
                # C2 / C3：学生用内置前端看原图并自回归 rollout
                with torch.no_grad():
                    front.eval()
                    vh_ng, cl_ng = front(b["pv"].to(dev), b["grid"], b["lens"])
                    vis_ng, vm_ng = tj.pad_vis([vh_ng[i, : cl_ng[i]].float() for i in range(len(cl_ng))], dev)
                    with rotation_context_with(registry, vis_ng, vm_ng,
                                               ans_start=[pad] * len(pids), text_mask=attn):
                        gen = model.generate(input_ids=in_ids, attention_mask=attn, max_new_tokens=args.max_new,
                                             min_new_tokens=args.min_new, do_sample=args.gen_temp > 0,
                                             temperature=max(args.gen_temp, 1e-5), top_p=0.9,
                                             pad_token_id=pad_id, eos_token_id=stops)
                    front.train()
                gids = [topd.cut_at_stop(gen[k, pad:], set(stops), include_stop=True) for k in range(len(idx))]

            imgs = [Image.open(paths[data["ids"][i]]).convert("RGB") for i in idx]
            enc, Lp, bad = topd.build_teacher_batch(proc, prompts, pids, gids, imgs, dev, pad_id)
            if ms == 1:
                assert sum(bad) == 0, f"师生轨迹 token 对齐检查失败：bad={bad}"
                log(f"TOKEN_ALIGN_CHECK passed on batch size {len(idx)} (bad=0, Lp={Lp})")
            with torch.no_grad():
                tlog = teacher(**enc).logits

        # 带梯度前向：PatchFront 产生活体 v̂
        v_hat, cl = front(b["pv"].to(dev), b["grid"], b["lens"])
        vis, vm = tj.pad_vis([v_hat[i, : cl[i]] for i in range(len(cl))], dev)

        if need_teacher:
            maxa = max([len(g) for g in gids] + [1])
            tail = torch.full((len(idx), maxa), pad_id, dtype=in_ids.dtype, device=dev)
            tmask = torch.zeros(len(idx), maxa, dtype=attn.dtype, device=dev)
            for k, g in enumerate(gids):
                if g:
                    tail[k, : len(g)] = torch.tensor(g, device=dev)
                    tmask[k, : len(g)] = 1
            seq_ids, smask = torch.cat([in_ids, tail], dim=1), torch.cat([attn, tmask], dim=1)
            pos = (smask.cumsum(-1) - 1).clamp(min=0)
            with rotation_context_with(registry, vis, vm, ans_start=[pad] * len(idx), text_mask=smask):
                s_lg = model(input_ids=seq_ids, attention_mask=smask, position_ids=pos).logits[:, pad - 1 : pad - 1 + maxa].float()

            T = args.kd_temp
            for k, g in enumerate(gids):
                if not g or bad[k]:
                    continue
                na = len(g)
                t_logp = F.log_softmax(tlog[k, Lp - 1 : Lp - 1 + na].float() / T, dim=-1)
                s_logp = F.log_softmax(s_lg[k, :na] / T, dim=-1)
                if args.kl_mode in ("fwd", "offkd"):
                    kl_terms.append((t_logp.exp() * (t_logp - s_logp)).sum(-1).mean())
                elif args.kl_mode == "rev":
                    s_p = s_logp.exp()
                    kl_terms.append((s_p * (s_logp - t_logp)).sum(-1).mean())
                elif args.kl_mode == "jsd":
                    t_p, s_p = t_logp.exp(), s_logp.exp()
                    m_logp = torch.log(0.5 * (t_p + s_p).clamp(min=1e-12))
                    kl_terms.append((0.5 * (t_p * (t_logp - m_logp)).sum(-1) + 0.5 * (s_p * (s_logp - m_logp)).sum(-1)).mean())
                elif args.kl_mode == "seqkd":
                    t_top = t_logp.argmax(dim=-1)
                    kl_terms.append(F.nll_loss(s_logp, t_top) / (T ** 2))

        kl_loss = torch.stack(kl_terms).mean() * (args.kd_temp ** 2) if kl_terms else torch.tensor(0.0, device=dev)
        loss = args.kl_weight * kl_loss if need_teacher else torch.tensor(0.0, device=dev)
        if args.ce_weight > 0:
            ids_g, mask_g, lab_g = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            with rotation_context_with(registry, vis, vm, ans_start=b["a0s"], text_mask=mask_g):
                lg_g = model(input_ids=ids_g, attention_mask=mask_g).logits.float()[:, :-1]
            ce = F.cross_entropy(lg_g.reshape(-1, lg_g.shape[-1]), lab_g[:, 1:].reshape(-1))
            loss = loss + args.ce_weight * ce
        if args.feat_weight > 0:
            fl, _ = feature_loss(v_hat, b["tgts"], cl)
            loss = loss + args.feat_weight * fl

        if bool(torch.isfinite(loss)) and loss.requires_grad:
            (loss / args.accum).backward()
        hist.append((float(loss.item()), len(kl_terms), sum(len(g) for g in gids) / max(len(gids), 1)))
        if micro < args.accum - 1:
            continue
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        if step % 25 == 0:
            h = hist[-25:]
            log(f"step {step} loss={sum(x[0] for x in h)/len(h):.4f} kl_used={sum(x[1] for x in h)/len(h):.1f} "
                f"len={sum(x[2] for x in h)/len(h):.1f}tok gn={float(gn):.2f} {(time.time()-t0)/step:.2f}s/it")

    result["train_seconds"] = time.time() - t0
    fin_ev = full_front_eval(model, front, registry, tok, data, val_eval, val_em_1000, paths, ip, dev, args.seq,
                             n_acc=args.n_acc, n_em=len(val_em_1000), max_new=args.max_new)
    result["eval"] = fin_ev
    result["delta"] = {k: fin_ev[k] - init_ev[k] for k in fin_ev if isinstance(fin_ev[k], float)}
    result["peak_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    log("FINAL[front v̂]: " + " ".join(f"{k}={v:.4f}" for k, v in fin_ev.items() if isinstance(v, (int, float))))
    log("DELTA[after - before]: " + " ".join(f"{k}={v:+.4f}" for k, v in result["delta"].items()))
    save_args = dict(ja)
    save_args["post_kl_mode"] = args.kl_mode
    save_args["post_kl_weight"] = args.kl_weight
    save_args["post_steps"] = args.steps
    save_args["seed"] = args.seed
    json.dump(result, open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
    torch.save({"front": front.state_dict(), "vrmg": [m.state_dict() for m in vrmgs], "args": save_args,
                "backbone": model.state_dict() if args.tune_backbone else None}, args.out + ".pt")
    log("JOINT_OPD_DONE")


if __name__ == "__main__":
    main()
