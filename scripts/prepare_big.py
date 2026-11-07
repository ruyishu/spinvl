"""万级数据的老师信号抽取（分片、可断点续跑、视觉特征走 memmap）。

每个分片产出 <prefix><k>.feats.bin（fp16，(总 token, vdim)）与 <prefix><k>.meta.pt（最后写入，作为完成标记）。
对齐方式与 prepare_final.py 完全一致：老师在第 p 位预测第 p+1 个词，答案段预测位置 a0-1..a1-2。
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

B = "/data/ibu-bi-ai-share-disk"
PICK_LAYERS = [8, 16, 24, 32]
VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TMPL = "Question: {q}\nAnswer:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--teacher", default="Qwen3-VL-8B-Instruct")
    ap.add_argument("--shard-size", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--max-text-tokens", type=int, default=240)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.limit:
        rows = rows[: args.limit]
    tpath = f"{B}/zk/model/{args.teacher}"
    proc = AutoProcessor.from_pretrained(tpath, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    tok = proc.tokenizer
    teacher = AutoModelForImageTextToText.from_pretrained(
        tpath, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    vision = getattr(teacher, "visual", None) or teacher.model.visual
    merge = getattr(vision.config, "spatial_merge_size", None) or getattr(vision.config, "merge_size", 2)

    n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
    print(f"{len(rows)} rows -> {n_shards} shards", flush=True)
    t_all = time.time()
    for k in range(n_shards):
        meta_path = f"{args.out_prefix}{k}.meta.pt"
        if os.path.exists(meta_path):
            print(f"shard {k} exists, skip", flush=True)
            continue
        bin_path = f"{args.out_prefix}{k}.feats.bin"
        part = rows[k * args.shard_size : (k + 1) * args.shard_size]
        out = {"ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [],
               "topk_ids": [], "topk_vals": [], "hidden": [[] for _ in PICK_LAYERS],
               "voff": [], "vlen": [], "vdim": None, "feat_bin": bin_path}
        skipped, n_tok = 0, 0
        t0 = time.time()
        with open(bin_path, "wb") as fbin:
            for i in range(0, len(part), args.batch):
                chunk = part[i : i + args.batch]
                imgs, texts, keep = [], [], []
                for r in chunk:
                    pt = PROMPT_TMPL.format(q=r["question"]) + " " + r["answer"]
                    if len(tok(pt)["input_ids"]) > args.max_text_tokens:
                        skipped += 1
                        continue
                    imgs.append(Image.open(os.path.join(args.img_root, r["image"])).convert("RGB"))
                    texts.append(VISION_PREFIX + pt)
                    keep.append(r)
                if not keep:
                    continue
                inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda:0")
                captured = {}

                def _hook(_m, _i, _o):
                    captured["v"] = _o[0] if isinstance(_o, (tuple, list)) else _o

                h = vision.register_forward_hook(_hook)
                try:
                    with torch.no_grad():
                        o = teacher(**inputs, output_hidden_states=True)
                finally:
                    h.remove()
                vf = captured["v"]
                L = inputs["input_ids"].shape[1]
                grid = inputs["image_grid_thw"].tolist()
                tcounts = [int(g[0] * g[1] * g[2] // (merge * merge)) for g in grid]
                if sum(tcounts) != vf.shape[0]:
                    raw = [g[0] * g[1] * g[2] for g in grid]
                    s = sum(raw)
                    tcounts = [int(round(x / s * vf.shape[0])) for x in raw]
                    tcounts[-1] += vf.shape[0] - sum(tcounts)
                out["vdim"] = int(vf.shape[-1])

                offset = 0
                for j, r in enumerate(keep):
                    pt = PROMPT_TMPL.format(q=r["question"]) + " " + r["answer"]
                    pids = tok(pt)["input_ids"]
                    aids = tok(" " + r["answer"])["input_ids"]
                    n_pt, n_a = len(pids), len(aids)
                    n_v = tcounts[j]
                    feat = vf[offset : offset + n_v]
                    offset += n_v
                    if L - n_pt < 1 or n_a == 0 or pids[-n_a:] != aids:
                        skipped += 1
                        continue
                    a0 = n_pt - n_a
                    base = L - n_pt
                    pos = torch.arange(base + a0 - 1, base + a0 - 1 + n_a, device="cuda:0")
                    top = torch.topk(o.logits[j, pos].float(), args.topk, dim=-1)
                    out["ids"].append(r["id"])
                    out["prompts"].append(PROMPT_TMPL.format(q=r["question"]))
                    out["answers"].append(r["answer"])
                    out["a_rel_start"].append(a0)
                    out["n_answer"].append(n_a)
                    out["topk_ids"].append(top.indices.to(torch.int32).cpu())
                    out["topk_vals"].append(top.values.to(torch.float16).cpu())
                    for li, l in enumerate(PICK_LAYERS):
                        out["hidden"][li].append(o.hidden_states[l][j, pos].to(torch.float16).cpu())
                    fbin.write(feat.to(torch.float16).cpu().numpy().tobytes())
                    out["voff"].append(n_tok)
                    out["vlen"].append(n_v)
                    n_tok += n_v
                if (i // args.batch) % 40 == 0:
                    done = len(out["ids"])
                    print(f"shard {k}: {done}/{len(part)} kept, skipped={skipped}, "
                          f"{done / max(time.time() - t0, 1):.1f} samp/s", flush=True)
        torch.save(out, meta_path)  # 最后写，作为分片完成标记
        print(f"SHARD_DONE {k}: {len(out['ids'])} samples, {n_tok} vision tokens, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"ALL_DONE {n_shards} shards in {(time.time() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
