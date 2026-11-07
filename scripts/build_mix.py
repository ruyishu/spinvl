"""把 DeepEyes 里的人工标注短答案 VQA（thinklite 推理 + chart 选择题）解出图片，造成 mix 数据集清单。

输出：$OUT/img/<id>.jpg 与 $OUT/manifest.jsonl（字段 id/question/answer/image，image 为相对 $OUT 的路径）。
图片缩放到 max_pixels 以内（按 32 的倍数取整，匹配 Qwen3-VL 的 patch16×merge2），减少视觉 token。
"""

import argparse
import glob
import io
import json
import os
from multiprocessing import Pool

import pyarrow.parquet as pq
from PIL import Image

ROOT = glob.glob("/data/ibu-bi-ai-share-disk/.cache/huggingface/hub/datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots/*")[0]
SOURCES = [("think", "data_thinklite_reasoning_acc.parquet"), ("chart", "data_v0.8_visual_toolbox_v2.parquet")]


def fit(img, max_pixels):
    w, h = img.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        w, h = int(w * s), int(h * s)
    w, h = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
    return img.resize((w, h), Image.BICUBIC)


def work(args):
    rid, raw, out, max_pixels = args
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = fit(img, max_pixels)
        p = f"{out}/img/{rid}.jpg"
        img.save(p, quality=92)
        return rid, f"img/{rid}.jpg", img.size
    except Exception:
        return rid, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-pixels", type=int, default=392 * 392)
    ap.add_argument("--max-ans-chars", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(f"{args.out}/img", exist_ok=True)

    rows, jobs = [], []
    for tag, fn in SOURCES:
        pf = pq.ParquetFile(f"{ROOT}/{fn}")
        n_src = 0
        for rg in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(rg, columns=["images", "reward_model", "extra_info"]).to_pylist():
                q = (r["extra_info"] or {}).get("question", "").strip()
                a = str((r["reward_model"] or {}).get("ground_truth", "")).strip()
                imgs = r["images"] or []
                if not q or not a or not imgs or len(a) > args.max_ans_chars or "\n" in a:
                    continue
                rid = f"{tag}_{n_src}"
                jobs.append((rid, imgs[0]["bytes"], args.out, args.max_pixels))
                rows.append({"id": rid, "question": q, "answer": a})
                n_src += 1
                if args.limit and n_src >= args.limit:
                    break
            if args.limit and n_src >= args.limit:
                break
        print(tag, "kept", n_src, flush=True)

    with Pool(12) as pool:
        res = {rid: (p, sz) for rid, p, sz in pool.imap_unordered(work, jobs, chunksize=64)}
    kept = 0
    with open(f"{args.out}/manifest.jsonl", "w") as f:
        for r in rows:
            p, sz = res.get(r["id"], (None, None))
            if p is None:
                continue
            r["image"] = p
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            kept += 1
    print("MANIFEST", kept, "->", f"{args.out}/manifest.jsonl")


if __name__ == "__main__":
    main()
