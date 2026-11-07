"""Geometry3K 准备：CLIP 特征 + 老师 top-k logits / 隐状态（答案段）。

对齐方式：plain_text = "Question: ...\\nAnswer: ..."，师生用**同一串文本**，
老师只在前面多挂图像 token，因此文本段逐 token 相同，位置只差一个偏移。
"""

import argparse
import json
import os

import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor, CLIPImageProcessor,
                          CLIPVisionModel)

B = "/data/ibu-bi-ai-share-disk"
TEACHER_ROOT = f"{B}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
CLIP_ROOT = f"{B}/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots"
PICK_LAYERS = [8, 16, 24, 32]
VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"


def snapshot(root):
    subs = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]
    return sorted(subs)[-1]


def plain_text(q, a):
    return f"Question: {q}\nAnswer: {a}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl)]
    print(f"{len(rows)} qa pairs", flush=True)

    tpath = snapshot(TEACHER_ROOT)
    proc = AutoProcessor.from_pretrained(tpath, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    tok = proc.tokenizer
    teacher = AutoModelForImageTextToText.from_pretrained(
        tpath, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()

    cpath = snapshot(CLIP_ROOT)
    cproc = CLIPImageProcessor.from_pretrained(cpath)
    clip = CLIPVisionModel.from_pretrained(cpath, dtype=torch.float16, device_map="cuda:0").eval()

    out = {"files": [], "topk_ids": [], "topk_vals": [], "hidden": [[], [], [], []], "feats": [],
           "texts": [], "a_rel_start": [], "n_answer": []}
    skipped = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        imgs, texts = [], []
        for r in chunk:
            imgs.append(Image.open(os.path.join(args.img_root, r["image"])).convert("RGB"))
            texts.append(VISION_PREFIX + plain_text(r["question"], r["answer"]))

        inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda:0")
        with torch.no_grad():
            o = teacher(**inputs, output_hidden_states=True)
            px = cproc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda:0", dtype=torch.float16)
            cfeat = clip(pixel_values=px).last_hidden_state.cpu()
        L = inputs["input_ids"].shape[1]

        for j, r in enumerate(chunk):
            pt = plain_text(r["question"], r["answer"])
            pids = tok(pt)["input_ids"]
            aids = tok(r["answer"])["input_ids"]
            n_pt, n_a = len(pids), len(aids)
            offset = L - n_pt  # 左填充后文本段的起点
            if offset < 0 or pids[-n_a:] != aids or offset < 1:
                skipped += 1
                continue
            a_start = offset + n_pt - n_a
            pos = torch.arange(a_start - 1, a_start - 1 + n_a, device="cuda:0")
            top = torch.topk(o.logits[j, pos].float(), args.topk, dim=-1)
            out["files"].append(r["id"])
            out["topk_ids"].append(top.indices.to(torch.int32).cpu())
            out["topk_vals"].append(top.values.to(torch.float16).cpu())
            for k, l in enumerate(PICK_LAYERS):
                out["hidden"][k].append(o.hidden_states[l][j, pos].to(torch.float16).cpu())
            out["feats"].append(cfeat[j])
            out["texts"].append(pt)
            out["a_rel_start"].append(n_pt - n_a)
            out["n_answer"].append(n_a)
        if (i // args.batch) % 20 == 0:
            print(f"{i+len(chunk)}/{len(rows)}  skipped={skipped}", flush=True)

    out["feats"] = torch.stack(out["feats"])
    torch.save(out, args.out)
    print(f"DONE {len(out['files'])} skipped={skipped} -> {args.out}")


if __name__ == "__main__":
    main()
