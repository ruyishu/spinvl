"""Geometry3K 三臂对照：ce / kd / rel。目标答案来自人工标注，不是老师生成。"""

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


def collate(batch, tok, max_seq, data, use_teacher):
    texts = [data["texts"][i] for i in batch]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_seq, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    T = max_seq
    for j, i in enumerate(batch):
        a0 = data["a_rel_start"][i]
        a1 = a0 + data["n_answer"][i]
        labels[j, :a0] = -100
        labels[j, a1:] = -100
    feats = torch.stack([data["feats"][i] for i in batch])

    k_ids = k_vals = k_mask = hidden = None
    if use_teacher:
        k = data["topk_ids"][batch[0]].shape[-1]
        k_ids = torch.zeros(len(batch), T, k, dtype=torch.long)
        k_vals = torch.zeros(len(batch), T, k)
        k_mask = torch.zeros(len(batch), T)
        hidden = [torch.zeros(len(batch), T, data["hidden"][li][batch[0]].shape[-1], dtype=torch.float16)
                  for li in range(len(data["hidden"]))]
        for j, i in enumerate(batch):
            n = data["n_answer"][i]
            a0 = data["a_rel_start"][i]
            k_ids[j, a0 - 1 : a0 - 1 + n] = data["topk_ids"][i].long()[:n]
            k_vals[j, a0 - 1 : a0 - 1 + n] = data["topk_vals"][i].float()[:n]
            k_mask[j, a0 - 1 : a0 - 1 + n] = 1.0
            for li in range(len(data["hidden"])):
                hidden[li][j, a0 - 1 : a0 - 1 + n] = data["hidden"][li][i][:n]
    return enc["input_ids"], enc["attention_mask"], labels, feats, k_ids, k_vals, k_mask, hidden


def evaluate(model, tok, val_idx, data, max_seq, bs, device, n_batches=8):
    model.eval()
    tot = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    cnt = 0
    with torch.no_grad():
        for bi in range(n_batches):
            batch = val_idx[bi * bs : (bi + 1) * bs]
            if len(batch) < 2:
                break
            ids, mask, labels, vf, *_ = collate(batch, tok, max_seq, data, False)
            ids, mask, labels, vf = ids.to(device), mask.to(device), labels.to(device), vf.to(device, torch.float32)
            perm = torch.tensor([(i + 1) % len(batch) for i in range(len(batch))])
            for key, vis in (("correct", vf), ("shuffled", vf[perm]), ("no_image", None)):
                if vis is None:
                    lg = model(input_ids=ids, attention_mask=mask).logits.float()
                else:
                    with rotation_context(vis):
                        lg = model(input_ids=ids, attention_mask=mask).logits.float()
                tot[key] += F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1)).item()
            cnt += 1
    model.train()
    return {k: v / max(cnt, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ce", "kd", "rel"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24")
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--query-dim", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--kd-weight", type=float, default=0.3)
    ap.add_argument("--rel-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    data = torch.load(args.data, map_location="cpu")
    n = len(data["files"])
    by_f = {}
    for idx_f, fname in enumerate(data["files"]):
        by_f.setdefault(str(fname), []).append(idx_f)
    fkeys = sorted(by_f)
    random.Random(0).shuffle(fkeys)
    n_val = max(64, n // 10)
    val_idx, train_idx = [], []
    for fk in fkeys:
        (val_idx if len(val_idx) < n_val else train_idx).extend(by_f[fk])
    if args.limit:
        train_idx = train_idx[: args.limit]
    print(f"arm={args.arm} train={len(train_idx)} val={len(val_idx)}", flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev, trust_remote_code=True)
    model.eval()
    layers = [int(x) for x in args.layers.split(",") if x]
    vrmgs = attach_rotation(model, layers, num_blocks=args.blocks, visual_dim=data["feats"].shape[-1],
                            query_dim=args.query_dim)
    params = [p for m in vrmgs for p in m.parameters()]
    n_train = freeze_all_but(model, params)
    print(f"vrmg={len(vrmgs)} trainable={n_train/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    logf = open(args.log, "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    log(f"=== qa {args.arm} steps={args.steps} bs={args.bs} lr={args.lr} kd={args.kd_weight} rel={args.rel_weight} ===")
    t0 = time.time()
    ce_hist, kd_hist, rel_hist = [], [], []
    for step in range(1, args.steps + 1):
        batch = [train_idx[(step * args.bs + i) % len(train_idx)] for i in range(args.bs)]
        ids, mask, labels, vf, k_ids, k_vals, k_mask, hidden = collate(batch, tok, args.seq, data, args.arm != "ce")
        ids, mask, labels, vf = ids.to(dev), mask.to(dev), labels.to(dev), vf.to(dev, torch.float32)
        with rotation_context(vf):
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=(args.arm == "rel"))
        lg = out.logits.float()
        ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), labels[:, 1:].reshape(-1))
        loss = ce
        kd_v = rel_v = 0.0
        if args.arm in ("kd", "rel"):
            k_ids_g, k_vals_g, k_mask_g = k_ids.to(dev), k_vals.to(dev, torch.float32), k_mask.to(dev)
            stu = lg.gather(-1, k_ids_g.clamp(min=0))
            stu_logp = F.log_softmax(stu, dim=-1)
            tch_logp = F.log_softmax(k_vals_g, dim=-1)
            m = k_mask_g.unsqueeze(-1)
            kd = ((tch_logp.exp() * (tch_logp - stu_logp)) * m).sum() / m.sum().clamp(min=1.0)
            loss = loss + args.kd_weight * kd
            kd_v = kd.item()
        if args.arm == "rel":
            terms = []
            for li, sl in enumerate(STUDENT_LAYERS):
                sh = F.normalize(out.hidden_states[sl].float(), dim=-1)
                th = F.normalize(hidden[li].to(dev).float(), dim=-1)
                gs, gt = sh @ sh.transpose(1, 2), th @ th.transpose(1, 2)
                mm = k_mask_g.unsqueeze(1) * k_mask_g.unsqueeze(2)
                terms.append((((gs - gt) ** 2) * mm).sum() / mm.sum().clamp(min=1.0))
            rel = torch.stack(terms).mean()
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
            ev = evaluate(model, tok, val_idx, data, args.seq, args.bs, dev)
            log(f"  eval@{step}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) +
                f" gap={ev['shuffled']-ev['correct']:+.4f} peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GiB")

    ev = evaluate(model, tok, val_idx, data, args.seq, args.bs, dev)
    log(f"FINAL qa_{args.arm}: " + " ".join(f"{k}={v:.4f}" for k, v in ev.items()) +
        f" gap={ev['shuffled']-ev['correct']:+.4f}")
    log("QA_DONE")


if __name__ == "__main__":
    main()
