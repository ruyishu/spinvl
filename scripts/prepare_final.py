"""准备 Geometry3K：老师 Qwen3-VL 的**自带视觉特征** + 答案段 top-k logits + 隐状态。

关键对齐修复：老师在第 p 位预测第 p+1 个词。答案词在第 a0..a1-1 位，
所以预测位置是 a0-1 .. a1-2，缓存与学生的 logits[a0-1+i] 一一对应。
"""

import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

B = "/data/ibu-bi-ai-share-disk"
TEACHER_ROOT = f"{B}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
PICK_LAYERS = [8, 16, 24, 32]
VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"
PROMPT_TMPL = "Question: {q}\nAnswer:"


def snapshot(root):
    subs = [os.path.join(root, s) for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))]
    return sorted(subs)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--teacher", default="Qwen3-VL-4B-Instruct")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl)]
    local = f"{B}/zk/model/{args.teacher}"
    if os.path.isdir(local) and os.path.exists(os.path.join(local, "config.json")):
        tpath = local
    else:
        tpath = snapshot(f"{B}/.cache/huggingface/hub/models--Qwen--{args.teacher}/snapshots")
    proc = AutoProcessor.from_pretrained(tpath, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    tok = proc.tokenizer
    teacher = AutoModelForImageTextToText.from_pretrained(
        tpath, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    vision = getattr(teacher, "visual", None) or teacher.model.visual
    print(f"{len(rows)} rows; vision dim probe...", flush=True)

    out = {"ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [],
           "topk_ids": [], "topk_vals": [], "hidden": [[], [], [], []], "vfeat": [], "vlen": []}
    skipped = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        imgs, texts = [], []
        for r in chunk:
            imgs.append(Image.open(os.path.join(args.img_root, r["image"])).convert("RGB"))
            texts.append(VISION_PREFIX + PROMPT_TMPL.format(q=r["question"]) + " " + r["answer"])

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

        # 视觉 token 数：按 grid_thw 与 merge_size 换算，若与总长不符则按面积比例兜底
        grid = inputs["image_grid_thw"].tolist()
        merge = getattr(vision.config, "spatial_merge_size", None) or getattr(vision.config, "merge_size", 2)
        tcounts = [int(g[0] * g[1] * g[2] // (merge * merge)) for g in grid]
        total_v = vf.shape[0]
        if sum(tcounts) != total_v:
            raw = [g[0] * g[1] * g[2] for g in grid]
            s = sum(raw)
            tcounts = [int(round(r / s * total_v)) for r in raw]
            tcounts[-1] += total_v - sum(tcounts)
        offset = 0
        vfeats = []
        for n in tcounts:
            vfeats.append(vf[offset : offset + n].to(torch.float16).cpu())
            offset += n

        for j, r in enumerate(chunk):
            pt = PROMPT_TMPL.format(q=r["question"]) + " " + r["answer"]
            pids = tok(pt)["input_ids"]
            aids = tok(" " + r["answer"])["input_ids"]
            n_pt, n_a = len(pids), len(aids)
            if L - n_pt < 1 or pids[-n_a:] != aids or n_a == 0:
                skipped += 1
                continue
            a0 = n_pt - n_a
            offset = L - n_pt  # 文本段在老师序列里的起点
            pos = torch.arange(offset + a0 - 1, offset + a0 - 1 + n_a, device="cuda:0")
            top = torch.topk(o.logits[j, pos].float(), args.topk, dim=-1)
            out["ids"].append(r["id"])
            out["prompts"].append(PROMPT_TMPL.format(q=r["question"]))
            out["answers"].append(r["answer"])
            out["a_rel_start"].append(a0)
            out["n_answer"].append(n_a)
            out["topk_ids"].append(top.indices.to(torch.int32).cpu())
            out["topk_vals"].append(top.values.to(torch.float16).cpu())
            for k, l in enumerate(PICK_LAYERS):
                out["hidden"][k].append(o.hidden_states[l][j, pos].to(torch.float16).cpu())
            out["vfeat"].append(vfeats[j])
            out["vlen"].append(vfeats[j].shape[0])
        if (i // args.batch) % 20 == 0:
            print(f"{i+len(chunk)}/{len(rows)} skipped={skipped}", flush=True)

    torch.save(out, args.out)
    # 老师上限：答案首 token 的 top-1 命中率（越高越有蒸馏价值）
    hit = tot = 0
    for i, a in enumerate(out["answers"]):
        gold = tok(" " + a)["input_ids"]
        n = min(len(gold), out["topk_ids"][i].shape[0])
        hit += sum(int(out["topk_ids"][i][k][0].item() == gold[k]) for k in range(n))
        tot += n
    print(f"DONE {len(out['ids'])} skipped={skipped} vision_dim={out['vfeat'][0].shape[-1]} "
          f"vlen avg={sum(out['vlen'])/len(out['vlen']):.0f} teacher_top1_hit={hit/max(tot,1):.3f} "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
