"""语言能力保持：原模型 vs「把读图前端并进权重」的各变体（GSM8K 生成 + ARC-Challenge 选择）。

我们的设计里文本通路不含任何改动（前端只作用于读图通路）→ 与原模型逐位相同；
这里把前端永久并入前 K 层权重（单一权重集、encoder-free 部署的做法）作为消融，量化「合并」对语言的伤害：
  rot ：W' = W · blockdiag(R)^T（角度保持）；  lora：W' = W + scale·(A·B)^T（加性）
"""

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from front import PatchFront  # noqa: E402
from wrap import attach_rotation  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"


def load_gsm8k(n):
    t = pq.read_table(f"{B}/zk/data/gsm8k/main/test-00000-of-00001.parquet").to_pylist()[:n]
    return [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in t]


def load_arc(n):
    rows = pq.read_table(f"{B}/zk/data/arc/ARC-Challenge/test-00000-of-00001.parquet").to_pylist()[:n]
    out = []
    for r in rows:
        ch = r["choices"]
        if isinstance(ch, str):
            import ast
            ch = ast.literal_eval(ch)
        labels, texts = list(ch["label"]), list(ch["text"])
        gold = labels.index(r["answerKey"]) if r["answerKey"] in labels else 0
        out.append((r["question"], texts, gold))
    return out


@torch.no_grad()
def eval_gsm8k(model, tok, data, bs=25, max_new=320):
    tok.padding_side = "left"
    correct = 0
    for i in range(0, len(data), bs):
        chunk = data[i : i + bs]
        msgs = [tok.apply_chat_template(
            [{"role": "user", "content": q + "\nSolve step by step. End with '#### <number>'."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for q, _ in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        for (q, gold), o in zip(chunk, out):
            txt = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"####\s*\$?(-?[\d,\.]+)", txt)
            pred = (m.group(1) if m else (re.findall(r"-?\d[\d,\.]*", txt) or [""])[-1]).replace(",", "").rstrip(".")
            try:
                correct += int(abs(float(pred) - float(gold)) < 1e-4)
            except ValueError:
                pass
    return correct / len(data)


@torch.no_grad()
def eval_arc(model, tok, data):
    letters = ["A", "B", "C", "D", "E"]
    ids = [tok(" " + l)["input_ids"][-1] for l in letters]
    correct = 0
    for q, texts, gold in data:
        opts = "\n".join(f"{letters[k]}. {t}" for k, t in enumerate(texts[:5]))
        prompt = f"Question: {q}\n{opts}\nAnswer:"
        x = torch.tensor([tok(prompt)["input_ids"]], device="cuda")
        lg = model(input_ids=x).logits[0, -1].float()
        pred = int(torch.stack([lg[ids[k]] for k in range(min(len(texts), 5))]).argmax())
        correct += int(pred == gold)
    return correct / len(data)


def merge_into(model, front, mode, K):
    """把前端并进前 K 层权重（原地修改）；返回用于还原的备份。"""
    backup = {}
    layers = model.model.layers
    names = [("self_attn", n) for n in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
            [("mlp", n) for n in ("gate_proj", "up_proj", "down_proj")]
    if mode in ("rot", "rots"):
        front.static.prepare()
    for li in range(K):
        for parent, n in names:
            lin = getattr(getattr(layers[li], parent), n).rot.base
            backup[(li, n)] = lin.weight.data.clone()
            W = lin.weight.data.float()
            if mode in ("rot", "rots"):
                R = front.static._R[(li, n)][0].float()  # (nb, b, b)
                nb, b = R.shape[0], R.shape[1]
                if mode == "rots":
                    d_vec = front.static.scale[f"{li}_{n}"].float()
                    W = W * d_vec.unsqueeze(0)
                Wb = W.reshape(W.shape[0], nb, b)
                Wn = torch.einsum("onk,njk->onj", Wb, R).reshape(W.shape)  # (W @ D)_block @ R^T
                lin.weight.data = Wn.to(lin.weight.dtype)
            else:
                key = f"{li}_{n}"
                delta = (front.static.A[key].float() @ front.static.Bm[key].float()).T * front.static.scale
                lin.weight.data = (W + delta).to(lin.weight.dtype)
    return backup


def restore(model, backup):
    layers = model.model.layers
    for (li, n), w in backup.items():
        parent = layers[li].self_attn if n.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else layers[li].mlp
        getattr(parent, n).rot.base.weight.data = w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", default=[], help="name=mode:ckpt，如 rot4=rot:/path/front.pt")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--gsm-n", type=int, default=250)
    ap.add_argument("--arc-n", type=int, default=500)
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 trust_remote_code=True).eval()
    attach_rotation(model, [], wrap_extra=list(range(args.K)))
    gsm, arc = load_gsm8k(args.gsm_n), load_arc(args.arc_n)
    res = {}
    t0 = time.time()
    res["base"] = {"gsm8k": eval_gsm8k(model, tok, gsm), "arc_c": eval_arc(model, tok, arc)}
    print("base", res["base"], f"{time.time()-t0:.0f}s", flush=True)
    for v in args.variant:
        name, rest = v.split("=", 1)
        mode, ck = rest.split(":", 1)
        c = torch.load(ck, map_location="cuda")
        fr = PatchFront(model, K=args.K, mode=mode, vdim=c["vdim"],
                        lora_rank=c["args"].get("lora_rank", 16)).to("cuda")
        fr.load_state_dict(c["front"])
        backup = merge_into(model, fr, mode, args.K)
        res[name] = {"gsm8k": eval_gsm8k(model, tok, gsm), "arc_c": eval_arc(model, tok, arc)}
        restore(model, backup)
        print(name, res[name], f"{time.time()-t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"K": args.K, "gsm_n": args.gsm_n, "arc_n": args.arc_n, "results": res}, open(args.out, "w"), indent=1)
    print("LANG_DONE")


if __name__ == "__main__":
    main()
