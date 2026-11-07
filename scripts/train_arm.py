"""三条臂统一训练脚本：ce（纯文本） / kd（+老师 logit 蒸馏） / rel（+隐状态关系蒸馏）。

三张卡各跑一臂，数据、老师信号、步数、超参完全一致，只差损失项。
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
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrap import attach_rotation, freeze_all_but, rotation_context  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
STUDENT_LAYERS = [6, 13, 20, 27]


def collate(batch, tok, max_seq, cache, feats, idx):
    texts = [r["caption"] for r, _ in batch]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_seq, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    vf = torch.stack([feats[i] for _, i in batch])

    k_ids, k_vals, k_mask, hidden = None, None, None, None
    if cache is not None:
        rows = [cache["by_file"][r["file"]] for r, _ in batch]
        T = max_seq
        k = cache["topk_ids"][rows[0]].shape[-1]
        k_ids = torch.zeros(len(rows), T, k, dtype=torch.long)
        k_vals = torch.zeros(len(rows), T, k)
        k_mask = torch.zeros(len(rows), T)
        hidden = [torch.zeros(len(rows), T, cache["hidden"][li][rows[0]].shape[-1], dtype=torch.float16)
                  for li in range(len(cache["hidden"]))]
        for j, ri in enumerate(rows):
            n = min(cache["topk_ids"][ri].shape[0], T)
            k_ids[j, :n] = cache["topk_ids"][ri][:n].long()
            k_vals[j, :n] = cache["topk_vals"][ri][:n].float()
            k_mask[j, :n] = 1.0
            for li in range(len(cache["hidden"])):
                hidden[li][j, :n] = cache["hidden"][li][ri][:n]
    return enc["input_ids"], enc["attention_mask"], labels, vf, k_ids, k_vals, k_mask, hidden


def eval_losses(model, tok, data, feats, max_seq, bs, device, n_batches=8):
    model.eval()
    tot = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    cnt = 0
    with torch.no_grad():
        for bi in range(n_batches):
            batch = data[bi * bs : (bi + 1) * bs]
            if len(batch) < 2:
                break
            ids, mask, labels, vf, *_ = collate(batch, tok, max_seq, None, feats, None)
            ids, mask, labels, vf = ids.to(device), mask.to(device), labels.to(device), vf.to(device, torch.float32)
            perm = torch.tensor([(i + 1) % len(batch) for i in range(len(batch))])
            for key, vis in (("correct", vf), ("shuffled", vf[perm]), ("no_image", None)):
                if vis is None:
                    logits = model(input_ids=ids, attention_mask=mask).logits.float()
                else:
                    with rotation_context(vis):
                        logits = model(input_ids=ids, attention_mask=mask).logits.float()
                tot[key] += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)).item()
            cnt += 1
    model.train()
    return {k: v / max(cnt, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ce", "kd", "rel"], required=True)
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--teacher-cache", default="")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24")
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--query-dim", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本，0=全部")
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kd-weight", type=float, default=0.5)
    ap.add_argument("--rel-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=300)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.captions)]
    blob = torch.load(args.feats, map_location="cpu")
    feats = blob["feats"]
    fidx = {f: i for i, f in enumerate(blob["files"])}
    data = [(r, fidx[r["file"]]) for r in rows if r["file"] in fidx]

    cache = None
    if args.arm in ("kd", "rel"):
        assert args.teacher_cache, "--teacher-cache required for kd/rel"
        raw = torch.load(args.teacher_cache, map_location="cpu")
        cache = {
            "topk_ids": raw["topk_ids"], "topk_vals": raw["topk_vals"],
            "hidden": raw["hidden"], "by_file": {f: i for i, f in enumerate(raw["files"])},
        }
        data = [(r, i) for r, i in data if r["file"] in cache["by_file"]]

    random.Random(0).shuffle(data)
    n_val = max(64, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    if args.limit > 0:
        train = train[: args.limit]
    print(f"arm={args.arm} train={len(train)} val={len(val)}", flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev, trust_remote_code=True)
    model.eval()
    layers = [int(x) for x in args.layers.split(",") if x]
    vrmgs = attach_rotation(model, layers, num_blocks=args.blocks, visual_dim=feats.shape[-1], query_dim=args.query_dim)
    params = [p for m in vrmgs for p in m.parameters()]
    n_train = freeze_all_but(model, params)
    print(f"vrmg modules={len(vrmgs)} trainable={n_train/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    )

    logf = open(args.log, "a")
    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"=== {args.arm} layers={layers} bs={args.bs} seq={args.seq} steps={args.steps} lr={args.lr} "
        f"kd={args.kd_weight} rel={args.rel_weight} ===")

    t0 = time.time()
    kd_hist, rel_hist, ce_hist = [], [], []
    for step in range(1, args.steps + 1):
        batch = [train[(step * args.bs + i) % len(train)] for i in range(args.bs)]
        ids, mask, labels, vf, k_ids, k_vals, k_mask, hidden = collate(batch, tok, args.seq, cache, feats, None)
        ids, mask, labels, vf = ids.to(dev), mask.to(dev), labels.to(dev), vf.to(dev, torch.float32)
        with rotation_context(vf):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=(args.arm == "rel"))
            logits = out.logits.float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        loss = ce
        kd_v = rel_v = 0.0
        if args.arm in ("kd", "rel"):
            k_ids_g, k_vals_g, k_mask_g = k_ids.to(dev), k_vals.to(dev, torch.float32), k_mask.to(dev)
            # teacher[t] 对应 student 位置 t-1
            stu = logits[:, :-1].gather(-1, k_ids_g[:, :-1].clamp(min=0)).float()
            stu_logp = F.log_softmax(stu, dim=-1)
            tch_logp = F.log_softmax(k_vals_g[:, :-1], dim=-1)
            m = k_mask_g[:, :-1].unsqueeze(-1)
            kd = (((tch_logp.exp() * (tch_logp - stu_logp)) * m).sum() / m.sum().clamp(min=1.0))
            loss = loss + args.kd_weight * kd
            kd_v = kd.item()
        if args.arm == "rel":
            hs = out.hidden_states
            rel_terms = []
            for li, sl in enumerate(STUDENT_LAYERS):
                sh = hs[sl][:, :, :].float()
                th = hidden[li].to(dev).float()
                shn = F.normalize(sh, dim=-1)
                thn = F.normalize(th, dim=-1)
                gs = shn @ shn.transpose(1, 2)
                gt = thn @ thn.transpose(1, 2)
                mm = (k_mask_g.unsqueeze(1) * k_mask_g.unsqueeze(2))
                rel_terms.append((((gs - gt) ** 2) * mm).sum() / mm.sum().clamp(min=1.0))
            rel = torch.stack(rel_terms).mean()
            loss = loss + args.rel_weight * rel
            rel_v = rel.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        ce_hist.append(ce.item())
        if kd_v: kd_hist.append(kd_v)
        if rel_v: rel_hist.append(rel_v)
        if step % 50 == 0:
            log(f"step {step} ce={sum(ce_hist[-50:])/50:.4f} kd={sum(kd_hist[-50:])/max(len(kd_hist[-50:]),1):.3f} "
                f"rel={sum(rel_hist[-50:])/max(len(rel_hist[-50:]),1):.4f} gn={gn.item():.1f} {(time.time()-t0)/step:.2f}s/it")
        if step % args.eval_every == 0:
            ev = eval_losses(model, tok, val, feats, args.seq, args.bs, dev)
            log(f"  eval@{step}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) +
                f" gap={ev['shuffled']-ev['correct']:+.4f} peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GiB")

    ev = eval_losses(model, tok, val, feats, args.seq, args.bs, dev)
    log(f"FINAL {args.arm}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) +
        f" gap={ev['shuffled']-ev['correct']:+.4f}")
    torch.save({"state": [m.state_dict() for m in vrmgs], "args": vars(args)},
               args.log.replace(".log", "_vrmg.pt"))
    log("ARM_DONE")


if __name__ == "__main__":
    main()
