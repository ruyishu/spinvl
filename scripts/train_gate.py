"""Step 1.2 门禁实验：旋转到底有没有把图片内容传进去？

训练：冻结学生 LLM，只训 VRMG，任务是看图写 caption（teacher forcing）。
判据三道：
  1) 正常配对 loss  <  无图（R=I）loss      → 旋转确实注入了视觉
  2) 正常配对 loss  <  打乱图片 loss         → 用的是「这张图」而不只是「有视觉模式」
  3) 训练 loss 单调下降                      → 优化链路正常
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


def load_data(captions, feats_path, max_seq):
    rows = [json.loads(l) for l in open(captions)]
    blob = torch.load(feats_path, map_location="cpu")
    idx = {f: i for i, f in enumerate(blob["files"])}
    keep = [(r, idx[r["file"]]) for r in rows if r["file"] in idx]
    return keep, blob["feats"]


def collate(batch, tok, max_seq):
    texts = [r["caption"] for r, _ in batch]
    enc = tok(texts, padding="max_length", truncation=True, max_length=max_seq, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    feats = torch.stack([blob_feats[i] for _, i in batch])
    return enc["input_ids"], enc["attention_mask"], labels, feats


def evaluate(model, tok, data, feats, max_seq, bs, device, n_batches=8, shuffle_seed=1234):
    model.eval()
    rng = random.Random(shuffle_seed)
    totals = {"correct": 0.0, "shuffled": 0.0, "no_image": 0.0}
    count = 0
    with torch.no_grad():
        for bi in range(n_batches):
            batch = data[bi * bs : (bi + 1) * bs]
            if len(batch) < 2:
                break
            ids, mask, labels, vf = collate(batch, tok, max_seq)
            ids, mask, labels, vf = ids.to(device), mask.to(device), labels.to(device), vf.to(device, torch.float32)
            perm = torch.tensor([(i + 1) % len(batch) for i in range(len(batch))])
            for key, vis in (("correct", vf), ("shuffled", vf[perm]), ("no_image", None)):
                if vis is None:
                    logits = model(input_ids=ids, attention_mask=mask).logits.float()
                else:
                    with rotation_context(vis):
                        logits = model(input_ids=ids, attention_mask=mask).logits.float()
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
                totals[key] += loss.item()
            count += 1
    model.train()
    return {k: v / max(count, 1) for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--blocks", type=int, default=32)
    ap.add_argument("--query-dim", type=int, default=128)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seq", type=int, default=96)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log", default="gate.log")
    args = ap.parse_args()

    global blob_feats
    data, blob_feats = load_data(args.captions, args.feats, args.seq)
    random.Random(0).shuffle(data)
    n_val = max(64, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    print(f"train={len(train)} val={len(val)} feat_dim={tuple(blob_feats.shape)}", flush=True)

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev, trust_remote_code=True)
    model.eval()
    layers = [int(x) for x in args.layers.split(",") if x]
    vrmgs = attach_rotation(model, layers, num_blocks=args.blocks, visual_dim=blob_feats.shape[-1], query_dim=args.query_dim)
    params = [p for m in vrmgs for p in m.parameters()]
    n_train = freeze_all_but(model, params)
    print(f"vrmg modules={len(vrmgs)} trainable={n_train/1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
    )

    logf = open(args.log, "a")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== start layers={layers} blocks={args.blocks} bs={args.bs} seq={args.seq} steps={args.steps} lr={args.lr} ===")
    ev0 = evaluate(model, tok, val, blob_feats, args.seq, args.bs, dev)
    log(f"step 0 eval: " + "  ".join(f"{k}={v:.4f}" for k, v in ev0.items()) + f"  gap(shuf-corr)={ev0['shuffled']-ev0['correct']:+.4f}")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = [train[(step * args.bs + i) % len(train)] for i in range(args.bs)]
        ids, mask, labels, vf = collate(batch, tok, args.seq)
        ids, mask, labels, vf = ids.to(dev), mask.to(dev), labels.to(dev), vf.to(dev, torch.float32)
        with rotation_context(vf):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0:
            log(f"step {step} loss={loss.item():.4f} gn={gn.item():.2f} lr={sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.2f}s/it")
        if step % args.eval_every == 0:
            ev = evaluate(model, tok, val, blob_feats, args.seq, args.bs, dev)
            log(f"  eval@{step}: " + "  ".join(f"{k}={v:.4f}" for k, v in ev.items()) +
                f"  gap(shuf-corr)={ev['shuffled']-ev['correct']:+.4f}  peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GiB")
    ev = evaluate(model, tok, val, blob_feats, args.seq, args.bs, dev)
    log(f"FINAL: " + "  ".join(f"{k}={v:.4f}" for k, v in ev.items()) +
        f"  gap(shuf-corr)={ev['shuffled']-ev['correct']:+.4f}")
    torch.save({"state": {i: m.state_dict() for i, m in enumerate(vrmgs)}, "args": vars(args)},
               os.path.join(os.path.dirname(args.log), "vrmg_gate.pt"))
    log("GATE_DONE")


if __name__ == "__main__":
    main()
