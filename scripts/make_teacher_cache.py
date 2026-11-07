"""离线缓存老师信号：每张图 + caption 下，老师的 top-k logits 与 4 个中间层隐状态。

对齐：老师走 chat 模板（图+指令 → assistant=caption），只取 caption 段的前 n_cap 个位置，
      n_cap 用 tokenizer(caption) 的纯文本长度，与学生输入逐 token 对齐。
"""

import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

B = "/data/ibu-bi-ai-share-disk"
TEACHER_ROOT = f"{B}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
PICK_LAYERS = [8, 16, 24, 32]
PROMPT = "Describe this image in one detailed sentence."


def snapshot(root):
    subs = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]
    return sorted(subs)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=64)
    args = ap.parse_args()

    path = snapshot(TEACHER_ROOT)
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    rows = [json.loads(l) for l in open(args.captions)]
    print(f"{len(rows)} captions, teacher layers {PICK_LAYERS}", flush=True)

    all_ids, all_vals, all_hidden, keep = [], [], [], []
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        imgs, texts_full, texts_prompt = [], [], []
        n_caps = []
        for r in chunk:
            imgs.append(Image.open(os.path.join(args.img_dir, r["file"])).convert("RGB"))
            texts_full.append(proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
                 {"role": "assistant", "content": [{"type": "text", "text": r["caption"]}]}],
                tokenize=False))
            texts_prompt.append(proc.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
                tokenize=False, add_generation_prompt=True))
            n_caps.append(len(tok(r["caption"], add_special_tokens=False)["input_ids"]))

        full = proc(text=texts_full, images=imgs, return_tensors="pt", padding=True).to("cuda:0")
        prompt = proc(text=texts_prompt, images=imgs, return_tensors="pt", padding=True)
        prefix = prompt["attention_mask"].sum(dim=1).tolist()
        n_full = full["attention_mask"].sum(dim=1).tolist()
        L = full["input_ids"].shape[1]

        with torch.no_grad():
            out = model(**full, output_hidden_states=True)
        logits = out.logits.float()
        hiddens = [out.hidden_states[l] for l in PICK_LAYERS]

        for j in range(len(chunk)):
            start = L - (n_full[j] - prefix[j])  # caption 起点
            n = min(n_caps[j], L - start)         # 与 tokenizer 纯文本口径对齐
            pos = torch.arange(start - 1, start - 1 + n, device=logits.device)
            top = torch.topk(logits[j, pos], args.topk, dim=-1)
            all_ids.append(top.indices.to(torch.int32).cpu())
            all_vals.append(top.values.to(torch.float16).cpu())
            all_hidden.append([h[j, pos].to(torch.float16).cpu() for h in hiddens])
        keep.extend([r["file"] for r in chunk])
        if (i // args.batch) % 20 == 0:
            print(f"{i+len(chunk)}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)

    torch.save(
        {
            "files": keep,
            "topk_ids": all_ids,
            "topk_vals": all_vals,
            "hidden": [[all_hidden[i][k] for i in range(len(all_hidden))] for k in range(len(PICK_LAYERS))],
            "pick_layers": PICK_LAYERS,
        },
        args.out,
    )
    print(f"DONE {len(keep)} -> {args.out}  {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
