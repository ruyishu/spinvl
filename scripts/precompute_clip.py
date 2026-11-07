"""用 CLIP ViT-L/14-336 预提视觉特征，存成 fp16 张量供训练直接索引。"""

import argparse
import json
import os

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel

B = "/data/ibu-bi-ai-share-disk"
CLIP = f"{B}/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots"


def snapshot(root):
    subs = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]
    return sorted(subs)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.captions)]
    path = snapshot(CLIP)
    proc = CLIPImageProcessor.from_pretrained(path)
    model = CLIPVisionModel.from_pretrained(path, dtype=torch.float16, device_map="cuda:0").eval()

    feats = []
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            imgs = [Image.open(os.path.join(args.img_dir, r["file"])).convert("RGB") for r in chunk]
            px = proc(images=imgs, return_tensors="pt")["pixel_values"].to("cuda:0", dtype=torch.float16)
            out = model(pixel_values=px).last_hidden_state  # (B, 576, 1024)
            feats.append(out.cpu())
            if i % (args.batch * 10) == 0:
                print(f"{i}/{len(rows)}", flush=True)
    feats = torch.cat(feats, dim=0)
    torch.save({"files": [r["file"] for r in rows], "feats": feats}, args.out)
    print(f"saved {tuple(feats.shape)} -> {args.out}")


if __name__ == "__main__":
    main()
