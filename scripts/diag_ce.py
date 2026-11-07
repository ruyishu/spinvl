"""诊断：加载 ce 臂权重，看生成输出长什么样，并算 teacher-forced 的首 token 命中率。"""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrap import attach_rotation, rotation_context_with  # noqa: E402

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
B = "/data/ibu-bi-ai-share-disk"


def main():
    data = torch.load(f"{D}/data/geo_final.pt", map_location="cpu")
    n = len(data["ids"])
    order = list(range(n))
    random.Random(0).shuffle(order)
    n_val = max(64, n // 10)
    val = order[:n_val]

    tok = AutoTokenizer.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    vrmgs, registry = attach_rotation(model, [0, 4, 8, 12, 16, 20, 24], block_size=64,
                                      visual_dim=2560, query_dim=256)
    ck = torch.load(f"{D}/logs/final_ce_pt.pt", map_location="cuda")
    for m, sd in zip(vrmgs, ck["state"]):
        m.load_state_dict(sd)
    for m in vrmgs:
        m.eval()

    # teacher-forced 首 token 命中
    hit = tot = 0
    with torch.no_grad():
        for i in val[:120]:
            text = data["prompts"][i] + " " + data["answers"][i]
            ids = tok(text)["input_ids"]
            a0, na = data["a_rel_start"][i], data["n_answer"][i]
            vis = data["vfeat"][i].unsqueeze(0).to("cuda").float()
            vm = torch.ones(1, vis.shape[1], device="cuda")
            with rotation_context_with(registry, vis, vm):
                lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()
            pred = lg[a0 - 1].argmax().item()
            hit += int(pred == ids[a0])
            tot += 1
    print(f"teacher-forced 首 token 命中率: {hit}/{tot} = {hit/tot:.3f}")

    print("\n--- 生成样例 ---")
    for i in val[:5]:
        ids = tok(data["prompts"][i])["input_ids"]
        vis = data["vfeat"][i].unsqueeze(0).to("cuda").float()
        vm = torch.ones(1, vis.shape[1], device="cuda")
        with torch.no_grad():
            with rotation_context_with(registry, vis, vm):
                out = model.generate(input_ids=torch.tensor([ids], device="cuda"), max_new_tokens=8,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
        pred = tok.decode(out[0, len(ids):], skip_special_tokens=True)
        print(f"Q: {data['prompts'][i][:60]!r}\n   gold={data['answers'][i]!r}  pred={pred!r}")


if __name__ == "__main__":
    main()
