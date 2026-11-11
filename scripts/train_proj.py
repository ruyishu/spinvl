"""同预算对照：标准 LLaVA 式「冻结视觉塔 + MLP Projector + 视觉 token 拼在文本前」。

作用：回答审稿人的第一个问题——同样的冻结 LLM、同样的数据、同样的步数，
传统 token 拼接和「旋转注入」到底谁准、谁省。视觉塔冻结、LLM 冻结，只训 projector。

基线性质：这里挂的是 8B 老师的**外部视觉塔**，视觉 token 全程进入 LLM 的全部层，
是「带外部视觉塔的传统拼接强基线」，用来对比准确率与 FLOPs 开销，不是「拔掉视觉塔」的对照。
切分协议与 train_final 完全相同（固定种子打乱全量 → 前 10% 作 val → --limit 只截训练集；答案后补 EOS），
所以 N=6000/21000 的 val 与 P2 的 ce 臂是同一批样本，CE / 首 token / 整答案准确率可以直接并排比较。
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
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
import train_final as tf  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
BASELINE_KIND = "external_vision_tower_token_concat"
BASELINE_NOTE = ("带外部视觉塔的传统拼接强基线：挂 8B 老师的冻结视觉塔，经 MLP Projector 变成视觉 token 拼在文本前，"
                 "视觉 token 全程进入 LLM 全部层；用来对比准确率与 FLOPs 开销，不是「拔掉视觉塔」的对照。")


class Proj(nn.Module):
    def __init__(self, din: int, d: int, hidden: int = 0):
        super().__init__()
        h = hidden or d
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--teacher", default=imgdata.TEACHER)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2250)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=0, help="CE 评测样本数，0=整个 val 集（与 P2 同口径）")
    ap.add_argument("--n-acc", type=int, default=0, help="teacher-forced 准确率样本数，0=整个 val 集")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gnd-weight", type=float, default=0.0, help="带梯度的同 batch 错图对比损失权重（>0 启用）")
    ap.add_argument("--gnd-margin", type=float, default=0.75, help="首答案 token 错图与真图 NLL 边际阈值（nat，0.5~1.0）")
    ap.add_argument("--fast-feats", action="store_true", help="直接读取 shard.feats.bin 中的预抽 8B 视觉特征（免实时过 ViT，速度提升 5x）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(args.out + ".log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    paths = {}
    for s in (args.source or []):
        paths.update(imgdata.parse_source(s)[1])
    data = tf.load_data(args.data)
    n = len(data["ids"])
    train, val, split_info = tf.split_train_val_by_image(data, paths if paths else None, seed=args.seed)
    if args.fast_feats or not paths:
        pass
    else:
        train = [i for i in train if data["ids"][i] in paths]
        val = [i for i in val if data["ids"][i] in paths]
    assert train and val, "按图切分后可用 train/val 为空"
    if args.limit:
        train = train[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.fast_feats:
        ip = vision = None
        vdim = int(data["vfeat"][0].shape[-1])
    else:
        proc = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)
        ip = proc.image_processor
        teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map=dev,
                                                              trust_remote_code=True).eval()
        vision = getattr(teacher, "visual", None) or teacher.model.visual
        for p in vision.parameters():
            p.requires_grad_(False)
        vdim = teacher.config.text_config.hidden_size
    d = model.config.hidden_size
    proj = Proj(vdim, d).to(dev).float()

    def collate(idx_list):
        pv, grids, ids, lab, imgs, a0s, vfeats = [], [], [], [], [], [], []
        for i in idx_list:
            if args.fast_feats:
                vfeats.append(data["vfeat"][i].float())
            else:
                img = Image.open(paths[data["ids"][i]]).convert("RGB")
                o = ip(images=[img], return_tensors="pt")
                pv.append(o["pixel_values"].to(torch.float16))
                grids.append(o["image_grid_thw"][0])
            text = data["prompts"][i] + " " + data["answers"][i] + (tok.eos_token or "")
            t = tok(text, truncation=True, max_length=args.seq)["input_ids"]
            a0 = data["a_rel_start"][i]
            if a0 >= len(t):
                t = tok(text, truncation=False, max_length=10**6)["input_ids"][:4096]
            l = [-100] * len(t)  # 必须在可能的重新分词之后再建，否则长度对不上会越界
            for k in range(a0, min(len(t), a0 + data["n_answer"][i] + 1)):  # +1：答案后的 EOS 一并监督（同 P2）
                l[k] = t[k]
            ids.append(torch.tensor(t)); lab.append(torch.tensor(l))
            a0s.append(min(max(1, a0), len(t) - 1))
            imgs.append(i)
        L = max(len(x) for x in ids)
        I = torch.full((len(ids), L), tok.pad_token_id or 0, dtype=torch.long)
        Lb = torch.full((len(ids), L), -100, dtype=torch.long)
        t_lens = [len(x) for x in ids]
        for k, (x, y) in enumerate(zip(ids, lab)):
            I[k, : len(x)] = x; Lb[k, : len(x)] = y
        if args.fast_feats:
            return dict(ids=I, lab=Lb, t_lens=t_lens, a0s=a0s, vfeats=vfeats, idx=imgs)
        Lv = max(x.shape[0] for x in pv)
        P = torch.zeros(len(pv), Lv, pv[0].shape[1], dtype=torch.float16)
        for k, x in enumerate(pv):
            P[k, : x.shape[0]] = x
        return dict(ids=I, lab=Lb, t_lens=t_lens, a0s=a0s, pv=P, grid=torch.stack(grids), lens=[x.shape[0] for x in pv], idx=imgs)

    @torch.no_grad()
    def encode(pv, grid, lens):
        rows = torch.cat([pv[k, : int(lens[k])] for k in range(len(lens))], dim=0)
        o = vision(rows.to(dev), grid_thw=grid.to(dev))
        vf = o[0] if isinstance(o, (tuple, list)) else o
        vf = vf.float()
        merge = getattr(vision.config, "spatial_merge_size", None) or getattr(vision.config, "merge_size", 2)
        counts = [int(g[0] * g[1] * g[2] // (merge * merge)) for g in grid.tolist()]
        if sum(counts) != vf.shape[0]:
            raw = [g[0] * g[1] * g[2] for g in grid.tolist()]
            s = sum(raw)
            counts = [int(round(r / s * vf.shape[0])) for r in raw]
            counts[-1] += vf.shape[0] - sum(counts)
        out, off = [], 0
        for c in counts:
            out.append(vf[off : off + c]); off += c
        return out

    def run_batch(b, img_mode="correct", return_p0=False):
        ids, lab = b["ids"].to(dev), b["lab"].to(dev)
        t_lens, a0s = b["t_lens"], b["a0s"]
        vs = [v.to(dev) for v in b["vfeats"]] if args.fast_feats else encode(b["pv"], b["grid"], b["lens"])
        if img_mode == "shuffled":
            vs = [vs[(k + 1) % len(vs)] for k in range(len(vs))]
        emb_all = model.get_input_embeddings()(ids)
        if img_mode != "no_image":
            pv_tok = [proj(v).to(emb_all.dtype) for v in vs]
            # 拼接 [v_k, text_k] 并左对齐/右 padding，彻底消除视觉 token 与文本 token 中间的 0-padding 空洞
            seq_embs, seq_labs, seq_masks, p0_list, y0_list = [], [], [], [], []
            for k in range(len(vs)):
                vk = pv_tok[k]
                tk = emb_all[k, : t_lens[k]]
                lk = lab[k, : t_lens[k]]
                cat_e = torch.cat([vk, tk], dim=0)
                cat_l = torch.cat([torch.full((vk.shape[0],), -100, device=dev, dtype=torch.long), lk], dim=0)
                seq_embs.append(cat_e)
                seq_labs.append(cat_l)
                p0_list.append(vk.shape[0] + a0s[k] - 1)
                y0_list.append(ids[k, a0s[k]])
            max_len = max(x.shape[0] for x in seq_embs)
            emb = torch.zeros(len(vs), max_len, d, device=dev, dtype=emb_all.dtype)
            lab_out = torch.full((len(vs), max_len), -100, device=dev, dtype=torch.long)
            mask = torch.zeros(len(vs), max_len, device=dev)
            for k, (ce_k, cl_k) in enumerate(zip(seq_embs, seq_labs)):
                L_k = ce_k.shape[0]
                emb[k, :L_k] = ce_k
                lab_out[k, :L_k] = cl_k
                mask[k, :L_k] = 1.0
            lab = lab_out
        else:
            emb = emb_all
            mask = (ids != (tok.pad_token_id or 0)).float()
            p0_list = [a0s[k] - 1 for k in range(len(vs))]
            y0_list = [ids[k, a0s[k]] for k in range(len(vs))]
        out = model(inputs_embeds=emb, attention_mask=mask).logits
        if return_p0:
            return out, lab, mask, torch.tensor(p0_list, device=dev), torch.stack(y0_list)
        return out, lab, mask

    log(f"=== proj baseline [{BASELINE_KIND}] N={len(train)} val={len(val)} steps={args.steps} "
        f"trainable={sum(p.numel() for p in proj.parameters())/1e6:.2f}M gnd_w={args.gnd_weight} ===")
    log(BASELINE_NOTE)
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.01)
    nw = 0 if args.fast_feats else args.workers
    dl = DataLoader(train, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=nw,
                    collate_fn=collate, persistent_workers=nw > 0)
    it, t0, hist, gnd_hist = iter(dl), time.time(), [], []
    for step in range(1, args.steps + 1):
        try:
            b = next(it)
        except StopIteration:
            it = iter(dl); b = next(it)
        f = min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        lg, lab, _, p0_r, y0_r = run_batch(b, "correct", return_p0=True)
        shift_lg, shift_lab = lg[:, :-1].reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1)
        ce = F.cross_entropy(shift_lg, shift_lab, ignore_index=-100)
        loss = ce
        if args.gnd_weight > 0.0 and b["ids"].shape[0] >= 2:
            lg_shuf, _, _, p0_s, _ = run_batch(b, "shuffled", return_p0=True)
            b_idx = torch.arange(lg.shape[0], device=dev)
            nll_real = F.cross_entropy(lg.float()[b_idx, p0_r], y0_r, reduction="none")
            nll_shuf = F.cross_entropy(lg_shuf.float()[b_idx, p0_s], y0_r, reduction="none")
            gnd = F.relu(args.gnd_margin - (nll_shuf - nll_real)).mean()
            loss = loss + args.gnd_weight * gnd
            gnd_hist.append(gnd.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
        opt.step()
        hist.append(ce.item())
        if step % 50 == 0:
            gnd_str = f" gnd={sum(gnd_hist[-50:])/len(gnd_hist[-50:]):.4f}" if gnd_hist else ""
            log(f"step {step} ce={sum(hist[-50:])/50:.4f}{gnd_str} gn={gn.item():.2f} {(time.time()-t0)/step:.2f}s/it")

    @torch.no_grad()
    def evaluate():
        res = {}
        dlv = DataLoader(val[: args.n_eval or len(val)], batch_size=args.bs, collate_fn=collate, num_workers=2)
        ce = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
        for b in dlv:
            for mode in ce:
                lg, lab, _ = run_batch(b, mode)
                ce[mode] += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1),
                                            ignore_index=-100).item()
        res.update({k: v / max(len(dlv), 1) for k, v in ce.items()})
        res["gap"] = res["shuffled"] - res["correct"]
        first = full = n = 0
        for i in val[: args.n_acc or len(val)]:
            text = data["prompts"][i] + " " + data["answers"][i]
            tid = tok(text)["input_ids"]
            a0, na = data["a_rel_start"][i], data["n_answer"][i]
            if args.fast_feats:
                v = data["vfeat"][i].float().to(dev)
            else:
                img = Image.open(paths[data["ids"][i]]).convert("RGB")
                o = ip(images=[img], return_tensors="pt")
                v = encode(o["pixel_values"].to(torch.float16).unsqueeze(0), o["image_grid_thw"],
                           [o["pixel_values"].shape[0]])[0]
            pe = proj(v).unsqueeze(0).to(model.dtype)
            emb = torch.cat([pe, model.get_input_embeddings()(torch.tensor([tid], device=dev))], dim=1)
            lg = model(inputs_embeds=emb).logits[0].float()
            off = pe.shape[1]
            preds = [lg[off + a0 - 1 + k].argmax().item() for k in range(na)]
            first += int(preds[0] == tid[a0]); full += int(preds == tid[a0 : a0 + na]); n += 1
        res["tf_first"] = first / max(n, 1)
        res["tf_full"] = full / max(n, 1)
        return res

    ev = evaluate()
    log("FINAL " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()))
    json.dump({"args": vars(args), "val": ev, "train_seconds": time.time() - t0,
               "peak_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
               "baseline_kind": BASELINE_KIND, "baseline_note": BASELINE_NOTE,
               "llm_layers_with_visual_tokens": model.config.num_hidden_layers,
               "n_train": len(train), "n_val": len(val)},
              open(args.out + ".json", "w"), indent=1, ensure_ascii=False)
    torch.save({"proj": proj.state_dict(), "args": vars(args)}, args.out + ".pt")
    log("PROJ_DONE")


if __name__ == "__main__":
    main()
