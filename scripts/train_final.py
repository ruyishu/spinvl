"""叙事 1 三臂对照（修好版）：ce / kd / rel，含对齐自检与真实生成准确率。

修掉的三处：
1) 位置对齐：老师的 top-k 存在「预测位置」，与学生的 logits[:, a0-1+i] 一一对应；
2) Top-k KL：先对学生全词表做 log_softmax，再 gather，并把词表外概率合成 rest 桶；
3) 关系蒸馏：逐位置余弦 + 可学习投影，单 token 答案也有梯度。
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
STUDENT_LAYERS = [6, 13, 20, 27]
TEACHER_LAYERS = [8, 16, 24, 32]


def collate(batch_idx, data, tok, max_seq, need_teacher):
    eos = tok.eos_token or ""
    eos_id = tok.convert_tokens_to_ids(eos) if eos else None
    texts = [data["prompts"][i] + " " + data["answers"][i] + eos for i in batch_idx]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_seq, return_tensors="pt")
    T = enc["input_ids"].shape[1]
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for j, i in enumerate(batch_idx):
        a0, n = data["a_rel_start"][i], data["n_answer"][i]
        end = a0 + n
        # 答案后面紧跟 EOS 时，把它也算作监督目标，模型才会学会停
        if eos_id is not None and end < T and enc["input_ids"][j, end].item() == eos_id:
            end += 1
        labels[j, :a0] = -100
        labels[j, end:] = -100

    vlen = max(data["vlen"][i] for i in batch_idx)
    dv = data["vfeat"][0].shape[-1]
    vis = torch.zeros(len(batch_idx), vlen, dv)
    vmask = torch.zeros(len(batch_idx), vlen)
    for j, i in enumerate(batch_idx):
        n = data["vlen"][i]
        vis[j, :n] = data["vfeat"][i].float()
        vmask[j, :n] = 1.0

    k_ids = k_vals = k_mask = hidden = None
    if need_teacher:
        k = data["topk_ids"][batch_idx[0]].shape[-1]
        k_ids = torch.zeros(len(batch_idx), T - 1, k, dtype=torch.long)
        k_vals = torch.zeros(len(batch_idx), T - 1, k)
        k_mask = torch.zeros(len(batch_idx), T - 1)
        hidden = [torch.zeros(len(batch_idx), T - 1, data["hidden"][li][batch_idx[0]].shape[-1],
                              dtype=torch.float16) for li in range(len(TEACHER_LAYERS))]
        for j, i in enumerate(batch_idx):
            a0, n = data["a_rel_start"][i], data["n_answer"][i]
            s, e = a0 - 1, a0 - 1 + n
            k_ids[j, s:e] = data["topk_ids"][i].long()[:n]
            k_vals[j, s:e] = data["topk_vals"][i].float()[:n]
            k_mask[j, s:e] = 1.0
            for li in range(len(TEACHER_LAYERS)):
                hidden[li][j, s:e] = data["hidden"][li][i][:n]
    return enc["input_ids"], enc["attention_mask"], labels, vis, vmask, k_ids, k_vals, k_mask, hidden


@torch.no_grad()
def eval_losses(model, registry, tok, val, data, max_seq, dev, bs=8, n_batches=8):
    model.eval()
    tot = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    cnt = 0
    for bi in range(n_batches):
        batch = val[bi * bs : (bi + 1) * bs]
        if len(batch) < 2:
            break
        ids, mask, labels, vis, vmask, *_ = collate(batch, data, tok, max_seq, False)
        ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
        vis, vmask = vis.to(dev), vmask.to(dev)
        perm = torch.tensor([(i + 1) % len(batch) for i in range(len(batch))])
        for key, v, vm in (("correct", vis, vmask), ("shuffled", vis[perm], vmask[perm]),
                           ("no_image", None, None)):
            if v is None:
                lg = model(input_ids=ids, attention_mask=mask).logits.float()
            else:
                with rotation_context_with(registry, v, vm):
                    lg = model(input_ids=ids, attention_mask=mask).logits.float()
            tot[key] += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                        labels[:, 1:].reshape(-1)).item()
        cnt += 1
    model.train()
    return {k: v / max(cnt, 1) for k, v in tot.items()}


def _norm(s):
    s = str(s).strip().lower()
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(r"\.0+$", "", s)
    return s


@torch.no_grad()
def eval_accuracy(model, registry, tok, val, data, dev, n=60, max_new=8):
    model.eval()
    hit = 0
    for i in val[:n]:
        prompt = data["prompts"][i]
        ids = tok(prompt)["input_ids"]
        ids_t = torch.tensor([ids], device=dev)
        vis = data["vfeat"][i].unsqueeze(0).to(dev).float()
        vm = torch.ones(1, vis.shape[1], device=dev)
        with rotation_context_with(registry, vis, vm):
            out = model.generate(input_ids=ids_t, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0, len(ids):], skip_special_tokens=True)
        if _norm(pred) == _norm(data["answers"][i]):
            hit += 1
    model.train()
    return hit / max(min(n, len(val)), 1)


class LazyFeats:
    """memmap 视觉特征：按样本取 (n_i, dim) 的 fp16 张量，避免整份进内存。"""

    def __init__(self, path, offsets, lens, dim):
        self.mm = np.memmap(path, dtype=np.float16, mode="r").reshape(-1, dim)
        self.off, self.lens = offsets, lens

    def __len__(self):
        return len(self.lens)

    def __getitem__(self, i):
        o, n = self.off[i], self.lens[i]
        return torch.from_numpy(np.array(self.mm[o : o + n]))


class MultiLazy:
    """多个分片的 LazyFeats 拼成一个按全局下标取样本的视图。"""

    def __init__(self, parts):
        self.parts = parts
        self.cum = np.cumsum([0] + [len(p) for p in parts])

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        k = int(np.searchsorted(self.cum, i, side="right") - 1)
        return self.parts[k][i - int(self.cum[k])]


def load_data(path):
    """兼容：单个 .pt（特征在内存）、单个 meta.pt（memmap）、或逗号分隔/通配的多分片 meta。"""
    import glob as _glob
    paths = []
    for p in path.split(","):
        paths += sorted(_glob.glob(p)) if any(c in p for c in "*?[") else [p]
    if len(paths) == 1:
        data = torch.load(paths[0], map_location="cpu")
        if "feat_bin" in data:
            data["vfeat"] = LazyFeats(data["feat_bin"], data["voff"], data["vlen"], data["vdim"])
        return data
    metas = [torch.load(p, map_location="cpu") for p in paths]
    merged = {k: [] for k in ("ids", "prompts", "answers", "a_rel_start", "n_answer", "topk_ids",
                              "topk_vals", "vlen")}
    merged["hidden"] = [[] for _ in metas[0]["hidden"]]
    lazies = []
    for m in metas:
        for k in merged:
            if k == "hidden":
                for li, h in enumerate(m["hidden"]):
                    merged["hidden"][li] += h
            else:
                merged[k] += m[k]
        lazies.append(LazyFeats(m["feat_bin"], m["voff"], m["vlen"], m["vdim"]) if "feat_bin" in m else m["vfeat"])
    merged["vfeat"] = MultiLazy(lazies)
    merged["vdim"] = metas[0]["vdim"]
    return merged


@torch.no_grad()
def eval_tf_acc(model, registry, tok, data, idx, dev):
    """teacher-forced：答案每个位置 argmax 是否等于金标；返回（首 token 准确率，整答案准确率）。"""
    first = full = 0
    for i in idx:
        ids = tok(data["prompts"][i] + " " + data["answers"][i])["input_ids"]
        a0, na = data["a_rel_start"][i], data["n_answer"][i]
        vis = data["vfeat"][i].unsqueeze(0).to(dev).float()
        vm = torch.ones(1, vis.shape[1], device=dev)
        with rotation_context_with(registry, vis, vm):
            lg = model(input_ids=torch.tensor([ids], device=dev)).logits[0].float()
        preds = [lg[a0 - 1 + k].argmax().item() for k in range(na)]
        first += int(preds[0] == ids[a0])
        full += int(preds == ids[a0 : a0 + na])
    n = max(len(idx), 1)
    return first / n, full / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ce", "kd", "rel"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--multifreq", action="store_true", help="RoPE 式高低频多粒度旋转 + 逐词频段门控")
    ap.add_argument("--tokrot", action="store_true", help="方案 1：文本词直接查图 + 免求逆逐词严格正交旋转")
    ap.add_argument("--hybrid", action="store_true", help="方案 2：浓缩 16 个空间锚点 token + 动态旋转混合")
    ap.add_argument("--prompt-alpha", type=float, default=1.0, help="Prompt 段旋转强度掩码（<1 保留原 LLM 读题能力）")
    ap.add_argument("--hyb-bound", type=float, default=0.0, help="hybrid 残差相对范数上限（>0 防长序列激活爆炸）")
    ap.add_argument("--qcond", action="store_true", help="问题条件化旋转：把题目文本池化向量叠到 VRMG queries")
    ap.add_argument("--direct-hyb", action="store_true", help="hybrid 直接在 256 维对全图视觉 cell 做无损交叉注意力")
    ap.add_argument("--query-dim", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=160)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--rel-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student-layers", default="6,13,20,27")
    ap.add_argument("--em-n", type=int, default=60)
    ap.add_argument("--test-data", default="")
    ap.add_argument("--test-em-n", type=int, default=150)
    ap.add_argument("--result-json", default="")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    data = load_data(args.data)
    n = len(data["ids"])
    order = list(range(n))
    random.Random(0).shuffle(order)  # 验证集划分固定，跨 seed 可比
    n_val = max(64, n // 10)
    val, train = order[:n_val], order[n_val:]
    if args.limit:
        train = train[: args.limit]
    random.Random(args.seed).shuffle(train)  # seed 只影响训练顺序与初始化
    torch.manual_seed(args.seed)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True)
    model.eval()
    layers = [int(x) for x in args.layers.split(",") if x]
    vrmgs, registry = attach_rotation(model, layers, block_size=args.block, multifreq=args.multifreq, tokrot=args.tokrot, hybrid=args.hybrid,
                                      visual_dim=data["vfeat"][0].shape[-1], query_dim=args.query_dim,
                                      prompt_alpha=args.prompt_alpha, hyb_bound=args.hyb_bound,
                                      qcond=args.qcond, direct_hyb=args.direct_hyb)
    params = [p for m in vrmgs for p in m.parameters()]
    rel_head = None
    if args.arm == "rel":
        d_s = model.config.hidden_size
        d_t = data["hidden"][0][0].shape[-1]
        rel_head = nn.Linear(d_s, d_t, bias=False).to(dev, dtype=torch.float32)
        params = params + list(rel_head.parameters())
    n_train = freeze_all_but(model, params)

    logf = open(args.log, "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"=== {args.arm} train={len(train)} val={len(val)} vrmg={len(vrmgs)} "
        f"trainable={n_train/1e6:.1f}M block={args.block} kd_w={args.kd_weight} T={args.kd_temp} ===")

    # 对齐自检：老师 top-1 命中 gold 的比例
    if args.arm != "ce":
        ids, mask, labels, vis, vmask, k_ids, k_vals, k_mask, _ = collate(train[:8], data, tok, args.seq, True)
        gold = labels[:, 1:].reshape(-1)
        top1 = k_ids[:, :, 0].reshape(-1)
        m = (gold != -100) & (k_mask.reshape(-1) > 0)
        hit = (gold[m] == top1[m]).float().mean().item()
        log(f"ALIGN_CHECK teacher_top1_hit={hit:.3f} on {int(m.sum())} tokens")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))

    t0 = time.time()
    ce_h, kd_h, rel_h = [], [], []
    for step in range(1, args.steps + 1):
        batch = [train[(step * args.bs + i) % len(train)] for i in range(args.bs)]
        ids, mask, labels, vis, vmask, k_ids, k_vals, k_mask, hidden = collate(
            batch, data, tok, args.seq, args.arm != "ce")
        ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
        vis, vmask = vis.to(dev), vmask.to(dev)
        with rotation_context_with(registry, vis, vmask):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=(args.arm == "rel"))
        lg = out.logits.float()[:, :-1]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1))
        loss, kd_v, rel_v = ce, 0.0, 0.0

        if args.arm in ("kd", "rel"):
            k_ids_g, k_vals_g, k_mask_g = k_ids.to(dev), k_vals.to(dev), k_mask.to(dev)
            full_logp = F.log_softmax(lg / args.kd_temp, dim=-1)
            stu_k = full_logp.gather(-1, k_ids_g.clamp(min=0))
            p_t = F.softmax(k_vals_g / args.kd_temp, dim=-1)
            p_rest = (1.0 - p_t.sum(-1)).clamp(min=1e-6)
            p_s_rest = (1.0 - stu_k.exp().sum(-1)).clamp(min=1e-6)
            kl_tok = (p_t * (torch.log(p_t.clamp(min=1e-9)) - stu_k)).sum(-1) \
                     + p_rest * (torch.log(p_rest) - torch.log(p_s_rest))
            m = k_mask_g
            kd = (kl_tok * m).sum() / m.sum().clamp(min=1.0) * (args.kd_temp ** 2)
            loss = loss + args.kd_weight * kd
            kd_v = kd.item()

        if args.arm == "rel":
            terms = []
            for li, sl in enumerate([int(x) for x in args.student_layers.split(',')]):
                hs = F.normalize(rel_head(out.hidden_states[sl][:, :-1].float()), dim=-1)
                th = F.normalize(hidden[li].to(dev).float(), dim=-1)
                cos = (hs * th).sum(-1)
                mm = k_mask_g
                terms.append(((1.0 - cos) * mm).sum() / mm.sum().clamp(min=1.0))
            rel = torch.stack(terms).mean()
            loss = loss + args.rel_weight * rel
            rel_v = rel.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        ce_h.append(ce.item())
        if kd_v: kd_h.append(kd_v)
        if rel_v: rel_h.append(rel_v)
        if step % 50 == 0:
            log(f"step {step} ce={sum(ce_h[-50:])/50:.4f} "
                f"kd={sum(kd_h[-50:])/max(len(kd_h[-50:]),1):.3f} "
                f"rel={sum(rel_h[-50:])/max(len(rel_h[-50:]),1):.4f} gn={gn.item():.1f} "
                f"{(time.time()-t0)/step:.2f}s/it")
        if step % args.eval_every == 0:
            ev = eval_losses(model, registry, tok, val, data, args.seq, dev)
            acc = eval_accuracy(model, registry, tok, val, data, dev, n=args.em_n)
            log(f"  eval@{step}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items())
                + f" gap={ev['shuffled']-ev['correct']:+.4f} EM={acc:.3f} "
                  f"peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GiB")

    ev = eval_losses(model, registry, tok, val, data, args.seq, dev, n_batches=len(val) // 8 + 1)
    acc = eval_accuracy(model, registry, tok, val, data, dev, n=args.em_n)
    v_first, v_full = eval_tf_acc(model, registry, tok, data, val, dev)
    log(f"FINAL {args.arm}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items())
        + f" gap={ev['shuffled']-ev['correct']:+.4f} EM={acc:.3f} tf_first={v_first:.3f} tf_full={v_full:.3f}")
    result = {"args": vars(args), "val": {**ev, "gap": ev["shuffled"] - ev["correct"], "em": acc,
                                          "tf_first": v_first, "tf_full": v_full},
              "train_steps": args.steps, "train_seconds": time.time() - t0,
              "peak_gib": torch.cuda.max_memory_allocated() / 1024**3}

    if args.test_data:
        tdata = load_data(args.test_data)
        tidx = list(range(len(tdata["ids"])))
        tev = eval_losses(model, registry, tok, tidx, tdata, args.seq, dev, n_batches=len(tidx) // 8 + 1)
        t_first, t_full = eval_tf_acc(model, registry, tok, tdata, tidx, dev)
        t_em = eval_accuracy(model, registry, tok, tidx, tdata, dev, n=args.test_em_n)
        result["test"] = {**tev, "gap": tev["shuffled"] - tev["correct"], "em": t_em,
                          "tf_first": t_first, "tf_full": t_full, "n": len(tidx)}
        log(f"TEST: " + " ".join(f"{k}={v:.4f}" for k, v in tev.items())
            + f" EM={t_em:.3f} tf_first={t_first:.3f} tf_full={t_full:.3f} n={len(tidx)}")

    ckpt = args.log.replace(".log", "_pt.pt")
    torch.save({"state": [m.state_dict() for m in vrmgs],
                "rel_head": rel_head.state_dict() if rel_head is not None else None,
                "args": vars(args)}, ckpt)
    if args.result_json:
        os.makedirs(os.path.dirname(args.result_json) or ".", exist_ok=True)
        json.dump(result, open(args.result_json, "w"), indent=1, ensure_ascii=False)
    log("FINAL_DONE")


if __name__ == "__main__":
    main()
