"""为 Geometry3K 训练集生成 8B 老师的简短 CoT（推理链 + \boxed{答案}），做成学生 CE 冷启动数据。

背景：Geometry3K 金标是裸数字，而 Qwen3-VL-8B-Instruct 的默认行为是先写推导。
在裸答案上做 OPD 会让师生目标打架（老师想把学生往长推理上拽），所以这里先构造
「推导 + \boxed{}」格式的冷启动集，之后 OPD 在学生的 CoT 轨迹上才有意义。

视觉特征直接复用 geo_8b.pt 的缓存（同一批图、同一个 8B 视觉塔），不重复跑塔。
输出与 train_final.load_data 兼容的单文件 .pt。
"""

import argparse
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

B = "/data/ibu-bi-ai-share-disk"
D = f"{B}/zk/project/rose-distill"
VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"
INST = ("Solve the geometry problem with a short derivation (2-4 steps), "
        "then output the final answer as \\boxed{...}.\n")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def norm(s):
    return re.sub(r"\s+", "", str(s)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=f"{B}/zk/project/mm-opsd/data/processed-mv/geometry3k_train.jsonl")
    ap.add_argument("--img-root", default=f"{B}/zk/project/mm-opsd/data/processed-mv")
    ap.add_argument("--base", default=f"{D}/data/geo_8b.pt")
    ap.add_argument("--out", default=f"{D}/data/geo_cot.pt")
    ap.add_argument("--teacher", default=f"{B}/zk/model/Qwen3-VL-8B-Instruct")
    ap.add_argument("--student-tok", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl)]
    if args.limit:
        rows = rows[: args.limit]
    base = torch.load(args.base, map_location="cpu")
    vmap = {rid: (base["vfeat"][i], base["vlen"][i]) for i, rid in enumerate(base["ids"])}
    rows = [r for r in rows if r["id"] in vmap]
    print(f"{len(rows)} rows with cached vfeat", flush=True)

    proc = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"
    teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map="cuda",
                                                          trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(args.student_tok, trust_remote_code=True)

    out = {"ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [], "vfeat": [], "vlen": []}
    n_correct = n_boxed = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        imgs, texts = [], []
        for r in chunk:
            imgs.append(Image.open(os.path.join(args.img_root, r["image"])).convert("RGB"))
            texts.append(VISION_PREFIX + INST + f"Question: {r['question']}\nAnswer:")
        enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = teacher.generate(**enc, max_new_tokens=args.max_new, do_sample=False,
                                   pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id)
        for j, r in enumerate(chunk):
            text = proc.tokenizer.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            m = BOXED_RE.findall(text)
            if m:
                n_boxed += 1
                n_correct += int(norm(m[-1]) == norm(r["answer"]))
            else:  # 没按格式收尾的，补上金标 boxed，保证冷启动数据格式一致
                text = text + f"\n\\boxed{{{r['answer']}}}"
            prompt = f"Question: {r['question']}\nAnswer:"
            ids_full = tok(prompt + " " + text)["input_ids"]
            aids = tok(" " + text)["input_ids"]
            vf, vl = vmap[r["id"]]
            out["ids"].append(r["id"])
            out["prompts"].append(prompt)
            out["answers"].append(text)
            out["a_rel_start"].append(len(ids_full) - len(aids))
            out["n_answer"].append(len(aids))
            out["vfeat"].append(vf)
            out["vlen"].append(vl)
        if (i // args.batch) % 5 == 0:
            print(f"{i + len(chunk)}/{len(rows)} boxed={n_boxed} teacher_correct={n_correct}", flush=True)

    torch.save(out, args.out)
    print(json.dumps({"n": len(out["ids"]), "boxed_rate": n_boxed / max(len(rows), 1),
                      "teacher_cot_acc": n_correct / max(len(rows), 1),
                      "mean_answer_tokens": sum(out["n_answer"]) / max(len(out["ids"]), 1)}, indent=1), flush=True)
    print("PREP_COT_DONE", flush=True)


if __name__ == "__main__":
    main()
