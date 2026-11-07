"""用 teacher-forced 首 token / 整段命中率，公平对比 ce / kd / rel 三个 checkpoint。"""

import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wrap import attach_rotation, rotation_context_with  # noqa: E402

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
B = "/data/ibu-bi-ai-share-disk"


@torch.no_grad()
def score(model, registry, tok, data, val, n=150):
    first_hit = full_hit = 0
    for i in val[:n]:
        text = data["prompts"][i] + " " + data["answers"][i]
        ids = tok(text)["input_ids"]
        a0, na = data["a_rel_start"][i], data["n_answer"][i]
        vis = data["vfeat"][i].unsqueeze(0).to("cuda").float()
        vm = torch.ones(1, vis.shape[1], device="cuda")
        with rotation_context_with(registry, vis, vm):
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()
        preds = [lg[a0 - 1 + k].argmax().item() for k in range(na)]
        first_hit += int(preds[0] == ids[a0])
        full_hit += int(preds == ids[a0 : a0 + na])
    return first_hit / n, full_hit / n


def main():
    data = torch.load(f"{D}/data/geo_final.pt", map_location="cpu")
    n = len(data["ids"])
    order = list(range(n))
    random.Random(0).shuffle(order)
    val = order[: max(64, n // 10)]

    tok = AutoTokenizer.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", trust_remote_code=True)
    for arm in ["ce", "kd", "rel"]:
        model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16,
                                                     device_map="cuda", trust_remote_code=True).eval()
        vrmgs, registry = attach_rotation(model, [0, 4, 8, 12, 16, 20, 24], block_size=64,
                                          visual_dim=2560, query_dim=256)
        ck = torch.load(f"{D}/logs/final_{arm}_pt.pt", map_location="cuda")
        for m, sd in zip(vrmgs, ck["state"]):
            m.load_state_dict(sd)
        for m in vrmgs:
            m.eval()
        f1, fa = score(model, registry, tok, data, val)
        # 无图基线
        registry_no = {k: v for k, v in registry.items()}
        print(f"{arm}: first-token acc = {f1:.3f}   full-answer acc = {fa:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

    # 无图（原模型）基线
    model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True).eval()
    f1 = fa = 0
    with torch.no_grad():
        for i in val[:150]:
            text = data["prompts"][i] + " " + data["answers"][i]
            ids = tok(text)["input_ids"]
            a0, na = data["a_rel_start"][i], data["n_answer"][i]
            lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0].float()
            preds = [lg[a0 - 1 + k].argmax().item() for k in range(na)]
            f1 += int(preds[0] == ids[a0])
            fa += int(preds == ids[a0 : a0 + na])
    print(f"no_image: first-token acc = {f1/150:.3f}   full-answer acc = {fa/150:.3f}")


if __name__ == "__main__":
    main()
