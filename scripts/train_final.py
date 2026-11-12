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


def get_image_key(i, data, paths=None, path_sha_cache=None):
    """返回样本 i 的规范化图像内容指纹（优先 img_sha1 -> 文件 SHA1 -> 视觉特征 SHA1）。"""
    import hashlib
    if path_sha_cache is None:
        path_sha_cache = {}
    img_sha_list = data.get("img_sha1")
    rid = data["ids"][i]
    if img_sha_list and i < len(img_sha_list) and img_sha_list[i]:
        return f"sha1:{img_sha_list[i]}"
    if paths and rid in paths:
        rp = os.path.realpath(paths[rid])
        if rp not in path_sha_cache:
            if os.path.exists(rp):
                path_sha_cache[rp] = "file_sha1:" + hashlib.sha1(open(rp, "rb").read()).hexdigest()
            else:
                path_sha_cache[rp] = f"path:{os.path.normpath(paths[rid])}"
        return path_sha_cache[rp]
    if "vfeat" in data and len(data["vfeat"]) == len(data["ids"]):
        vf = data["vfeat"][i]
        h = hashlib.sha1()
        h.update(str(tuple(vf.shape)).encode())
        h.update(np.ascontiguousarray(vf[:4].numpy()).tobytes())
        h.update(np.ascontiguousarray(vf[-4:].numpy()).tobytes())
        return f"vfeat_sha1:{h.hexdigest()}"
    return f"__sample__{rid}"


def build_index_derangement(indices, data, paths=None, seed=20261112):
    """对给定样本下标序列 indices 构造严格异图置换 shuf_indices（保证 100% 异图，same_cnt == 0）。"""
    import hashlib
    from collections import defaultdict
    n = len(indices)
    if n <= 1:
        return list(indices), n, ""
    cache = {}
    keys = [get_image_key(i, data, paths, cache) for i in indices]
    groups = defaultdict(list)
    for pos, k in enumerate(keys):
        groups[k].append(pos)
    if len(groups) <= 1:
        return list(indices), n, ""
    rng = random.Random(seed)
    group_list = [list(g) for g in sorted(groups.values(), key=lambda g: keys[g[0]])]
    for g in group_list:
        rng.shuffle(g)
    rng.shuffle(group_list)
    max_g = max(len(g) for g in group_list)
    assert max_g * 2 <= n, f"验证集最大同图组 {max_g} 超过总样本数一半 {n}，无法构造零碰撞异图置换"
    ordered = [pos for g in group_list for pos in g]
    shift = rng.randint(max_g, n - max_g)
    perm_pos = [0] * n
    for p_idx, orig_pos in enumerate(ordered):
        perm_pos[orig_pos] = ordered[(p_idx + shift) % n]
    shuf_indices = [indices[perm_pos[pos]] for pos in range(n)]
    same_cnt = sum(1 for pos in range(n) if keys[perm_pos[pos]] == keys[pos])
    assert same_cnt == 0, f"内部异图置换仍存在 {same_cnt} 对同图样本！"
    perm_sha1 = hashlib.sha1(json.dumps(shuf_indices).encode()).hexdigest()
    return shuf_indices, same_cnt, perm_sha1


def select_stratified_val_1000(val, data, n_target=1000, seed=0):
    """按任务来源前缀从验证集分层抽取固定 1000 题，返回 (selected_indices, sha1)。"""
    import hashlib
    from collections import defaultdict
    if len(val) <= n_target:
        sha = hashlib.sha1(json.dumps(list(val)).encode()).hexdigest()
        return list(val), sha
    by_src = defaultdict(list)
    for i in val:
        rid = str(data["ids"][i])
        src = rid.split(":")[0].split("_")[0]
        by_src[src].append(i)
    rng = random.Random(seed)
    srcs = sorted(by_src.keys())
    for s in srcs:
        by_src[s] = list(by_src[s])
        rng.shuffle(by_src[s])
    total = len(val)
    picked = []
    rem = []
    for s in srcs:
        bucket = by_src[s]
        k = max(1, int(round(len(bucket) * n_target / total)))
        k = min(k, len(bucket))
        picked.extend(bucket[:k])
        rem.extend(bucket[k:])
    rng.shuffle(rem)
    if len(picked) < n_target:
        picked.extend(rem[: n_target - len(picked)])
    elif len(picked) > n_target:
        picked = picked[:n_target]
    sha = hashlib.sha1(json.dumps(picked).encode()).hexdigest()
    return picked, sha


def build_inbatch_diff_image_perm(batch, data, paths=None):
    """为 batch 内每个样本寻找严格不同图像的负例下标；返回 (perm_tensor, valid_mask_tensor)。"""
    n = len(batch)
    cache = {}
    keys = [get_image_key(i, data, paths, cache) for i in batch]
    perm, valid = [], []
    for j in range(n):
        found = -1
        for step in range(1, n):
            cand = (j + step) % n
            if keys[cand] != keys[j]:
                found = cand
                break
        if found >= 0:
            perm.append(found)
            valid.append(1.0)
        else:
            perm.append(j)
            valid.append(0.0)
    return perm, valid


def split_train_val_by_image(data, paths=None, n_val=0, seed=0):
    """按图像内容哈希（优先 img_sha1 -> 真实图片文件字节 SHA1 -> 视觉特征 SHA1）分组切分，确保同图绝不跨 train/val。"""
    import hashlib
    n = len(data["ids"])
    n_val = n_val or max(64, n // 10)
    groups = {}
    path_sha_cache = {}
    for i in range(n):
        key = get_image_key(i, data, paths, path_sha_cache)
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    random.Random(0).shuffle(keys)
    val, train, val_groups, train_groups = [], [], set(), set()
    for key in keys:
        if len(val) < min(n_val, n - 1):
            val.extend(groups[key])
            val_groups.add(key)
        else:
            train.extend(groups[key])
            train_groups.add(key)
    assert train and val, "按图切分后 train 或 val 为空"
    assert not (train_groups & val_groups), "train/val 存在共享图片组"
    random.Random(seed).shuffle(train)
    split_sha1 = hashlib.sha1(json.dumps({"val": val, "n_train": len(train)}).encode()).hexdigest()
    return train, val, {"train_rows": len(train), "val_rows": len(val),
                        "train_groups": len(train_groups), "val_groups": len(val_groups),
                        "group_overlap": len(train_groups & val_groups),
                        "split_sha1": split_sha1}


def topk_conditional_kl(student_logits, topk_ids, topk_vals, mask, temperature=2.0):
    """top-k 截断教师分布与学生同一候选集上的条件 KL，不含伪造的 rest 概率桶。"""
    s_logits = student_logits.gather(-1, topk_ids.clamp(min=0)) / temperature
    t_logits = topk_vals / temperature
    s_logp = F.log_softmax(s_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0) * (temperature ** 2)


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
def eval_losses(model, registry, tok, val, data, max_seq, dev, bs=8, n_batches=8, shuf_seed=20261112):
    model.eval()
    tot = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    cnt = 0
    eval_slice = list(val[: n_batches * bs])
    shuf_slice, same_cnt, perm_sha1 = build_index_derangement(eval_slice, data, seed=shuf_seed)
    assert same_cnt == 0, f"eval_losses 异图置换存在 {same_cnt} 对同图！"
    for bi in range(n_batches):
        batch = eval_slice[bi * bs : (bi + 1) * bs]
        shuf_batch = shuf_slice[bi * bs : (bi + 1) * bs]
        if len(batch) < 2:
            break
        ids, mask, labels, vis, vmask, *_ = collate(batch, data, tok, max_seq, False)
        _, _, _, vis_shuf, vmask_shuf, *_ = collate(shuf_batch, data, tok, max_seq, False)
        ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
        vis, vmask = vis.to(dev), vmask.to(dev)
        vis_shuf, vmask_shuf = vis_shuf.to(dev), vmask_shuf.to(dev)
        a0_list = [data["a_rel_start"][i] for i in batch]
        for key, v, vm in (("correct", vis, vmask), ("shuffled", vis_shuf, vmask_shuf),
                           ("no_image", None, None)):
            if v is None:
                lg = model(input_ids=ids, attention_mask=mask).logits.float()
            else:
                with rotation_context_with(registry, v, vm, ans_start=a0_list, text_mask=mask):
                    lg = model(input_ids=ids, attention_mask=mask).logits.float()
            tot[key] += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]),
                                        labels[:, 1:].reshape(-1)).item()
        cnt += 1
    model.train()
    out = {k: v / max(cnt, 1) for k, v in tot.items()}
    out["shuf_same_image_count"] = same_cnt
    out["shuf_perm_sha1"] = perm_sha1
    return out


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
                              "topk_vals", "vlen", "img_sha1")}
    merged["hidden"] = [[] for _ in metas[0]["hidden"]]
    lazies = []
    for m in metas:
        for k in merged:
            if k == "hidden":
                for li, h in enumerate(m["hidden"]):
                    merged["hidden"][li] += h
            elif k == "img_sha1":
                merged["img_sha1"] += m.get("img_sha1", [None] * len(m["ids"]))
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
        with rotation_context_with(registry, vis, vm, ans_start=[a0], text_mask=torch.ones(1, len(ids), device=dev)):
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
    ap.add_argument("--no-rot", action="store_true", help="A1 消融：旋转恒等映射 R=I，仅保留视觉 cross-attention")
    ap.add_argument("--query-dim", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=160)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--rel-weight", type=float, default=0.1)
    ap.add_argument("--gnd-weight", type=float, default=0.0, help="带梯度的同 batch 错图对比损失权重（>0 启用）")
    ap.add_argument("--gnd-margin", type=float, default=0.75, help="首答案 token 错图与真图 NLL 边际阈值（nat，建议 0.5~1.0）")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--student-layers", default="6,13,20,27")
    ap.add_argument("--em-n", type=int, default=1000)
    ap.add_argument("--test-data", default="")
    ap.add_argument("--test-em-n", type=int, default=150)
    ap.add_argument("--init-ckpt", default="", help="可选：从已有 VRMG 权重热启动继续微调")
    ap.add_argument("--result-json", default="")
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    data = load_data(args.data)
    if args.arm in ("kd", "rel") and ("mix_v2" in args.data or "nat_supp" in args.data):
        raise ValueError("mix_v2 / nat_supp 改写了题干与选项顺序，不能直接跑依赖旧 token 对齐的离线 kd/rel 臂！")
    n = len(data["ids"])
    train, val, split_info = split_train_val_by_image(data, None, seed=args.seed)
    if args.limit:
        train = train[: args.limit]
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
                                      qcond=args.qcond, direct_hyb=args.direct_hyb, no_rot=args.no_rot)
    if args.init_ckpt and os.path.exists(args.init_ckpt):
        ick = torch.load(args.init_ckpt, map_location=dev)
        for m, sd in zip(vrmgs, ick["state"]):
            m.load_state_dict(sd, strict=False)
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

    log(f"=== {args.arm} train={len(train)} val={len(val)} vrmg={len(vrmgs)} split={split_info} "
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
    ce_h, kd_h, rel_h, gnd_h = [], [], [], []
    for step in range(1, args.steps + 1):
        batch = [train[(step * args.bs + i) % len(train)] for i in range(args.bs)]
        ids, mask, labels, vis, vmask, k_ids, k_vals, k_mask, hidden = collate(
            batch, data, tok, args.seq, args.arm != "ce")
        ids, mask, labels = ids.to(dev), mask.to(dev), labels.to(dev)
        vis, vmask = vis.to(dev), vmask.to(dev)
        a0_list = [data["a_rel_start"][i] for i in batch]
        with rotation_context_with(registry, vis, vmask, ans_start=a0_list, text_mask=mask):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=(args.arm == "rel"))
        lg = out.logits.float()[:, :-1]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1))
        loss, kd_v, rel_v, gnd_v = ce, 0.0, 0.0, 0.0

        if args.gnd_weight > 0.0 and len(batch) >= 2:
            perm_list, valid_list = build_inbatch_diff_image_perm(batch, data)
            if any(v > 0 for v in valid_list):
                perm = torch.tensor(perm_list, device=dev)
                valid_m = torch.tensor(valid_list, device=dev)
                with rotation_context_with(registry, vis[perm], vmask[perm], ans_start=a0_list, text_mask=mask):
                    out_shuf = model(input_ids=ids, attention_mask=mask)
                lg_shuf = out_shuf.logits.float()[:, :-1]
                p0_list = []
                for j, i in enumerate(batch):
                    a0 = data["a_rel_start"][i]
                    if a0 + 1 < ids.shape[1] and int(ids[j, a0].item()) == 220:
                        a0 = a0 + 1
                    p0_list.append(min(max(0, a0 - 1), lg.shape[1] - 1))
                p0 = torch.tensor(p0_list, device=dev)
                b_idx = torch.arange(len(batch), device=dev)
                y0 = ids[b_idx, p0 + 1]
                nll_real = F.cross_entropy(lg[b_idx, p0], y0, reduction="none")
                nll_shuf = F.cross_entropy(lg_shuf[b_idx, p0], y0, reduction="none")
                gnd_raw = F.relu(args.gnd_margin - (nll_shuf - nll_real))
                gnd = (gnd_raw * valid_m).sum() / valid_m.sum().clamp(min=1.0)
                loss = loss + args.gnd_weight * gnd
                gnd_v = gnd.item()

        if args.arm in ("kd", "rel"):
            k_ids_g, k_vals_g, k_mask_g = k_ids.to(dev), k_vals.to(dev), k_mask.to(dev)
            kd = topk_conditional_kl(lg, k_ids_g, k_vals_g, k_mask_g, args.kd_temp)
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
        if gnd_v: gnd_h.append(gnd_v)
        if step % 50 == 0:
            log(f"step {step} ce={sum(ce_h[-50:])/50:.4f} "
                f"kd={sum(kd_h[-50:])/max(len(kd_h[-50:]),1):.3f} "
                f"rel={sum(rel_h[-50:])/max(len(rel_h[-50:]),1):.4f} "
                f"gnd={sum(gnd_h[-50:])/max(len(gnd_h[-50:]),1):.4f} gn={gn.item():.1f} "
                f"{(time.time()-t0)/step:.2f}s/it")
        if step % args.eval_every == 0:
            ev = eval_losses(model, registry, tok, val, data, args.seq, dev)
            acc = eval_accuracy(model, registry, tok, val, data, dev, n=min(100, args.em_n))
            log(f"  eval@{step}: correct={ev['correct']:.4f} shuffled={ev['shuffled']:.4f} no_image={ev['no_image']:.4f}"
                + f" gap={ev['shuffled']-ev['correct']:+.4f} EM={acc:.3f} "
                  f"peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GiB")

    val_em_sub, val_1000_sha1 = select_stratified_val_1000(val, data, n_target=args.em_n, seed=0)
    ev = eval_losses(model, registry, tok, val, data, args.seq, dev, n_batches=len(val) // 8 + 1)
    acc = eval_accuracy(model, registry, tok, val_em_sub, data, dev, n=len(val_em_sub))
    v_first, v_full = eval_tf_acc(model, registry, tok, data, val, dev)
    log(f"FINAL {args.arm}: correct={ev['correct']:.4f} shuffled={ev['shuffled']:.4f} no_image={ev['no_image']:.4f}"
        + f" gap={ev['shuffled']-ev['correct']:+.4f} EM={acc:.3f}(n={len(val_em_sub)}) tf_first={v_first:.3f} tf_full={v_full:.3f}")
    result = {"args": vars(args), "split_info": split_info,
              "trainable_params": int(n_train),
              "val": {**ev, "gap": ev["shuffled"] - ev["correct"], "em": acc, "em_n": len(val_em_sub),
                      "val_em_sha1": val_1000_sha1, "tf_first": v_first, "tf_full": v_full},
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
