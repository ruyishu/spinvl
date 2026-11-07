"""用老师 MLLM 给 COCO 图生成 caption（ROSE S2 的数据级蒸馏做法）。

用法：python scripts/make_captions.py --img-dir ... --out ... --n 2500 --batch 8
可断点续跑：已有的 file 会跳过。
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


def snapshot(root):
    subs = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]
    return sorted(subs)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="/home/powerop/coco_data/train2017")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompt", default="Describe this image in one detailed sentence.")
    args = ap.parse_args()

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass
    print(f"resume: {len(done)} captions already exist")

    files = [f for f in sorted(os.listdir(args.img_dir)) if f.lower().endswith(".jpg")][: args.n]
    todo = [f for f in files if f not in done]
    print(f"todo {len(todo)} / {len(files)}")

    path = snapshot(TEACHER_ROOT)
    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    t0 = time.time()
    written = 0
    with open(args.out, "a") as fout:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i : i + args.batch]
            imgs, texts = [], []
            for name in chunk:
                imgs.append(Image.open(os.path.join(args.img_dir, name)).convert("RGB"))
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
                texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            inputs = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda:0")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
            gen = out[:, inputs["input_ids"].shape[1] :]
            caps = proc.batch_decode(gen, skip_special_tokens=True)
            for name, cap in zip(chunk, caps):
                fout.write(json.dumps({"file": name, "caption": cap.strip()}, ensure_ascii=False) + "\n")
            fout.flush()
            written += len(chunk)
            if (i // args.batch) % 10 == 0:
                rate = written / max(time.time() - t0, 1e-6)
                print(f"{written}/{len(todo)}  {rate:.1f} img/s  eta {(len(todo)-written)/max(rate,1e-6)/60:.1f} min", flush=True)
    print(f"DONE {written} captions -> {args.out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
