import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
B = "/data/ibu-bi-ai-share-disk"
tok = AutoTokenizer.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", trust_remote_code=True)
data = torch.load(f"{D}/data/geo_final.pt", map_location="cpu")
model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True).eval()

idx = list(range(8))
for tag, suffix in (("no_eos", ""), ("with_eos", tok.eos_token)):
    texts = [data["prompts"][i] + " " + data["answers"][i] + suffix for i in idx]
    enc = tok(texts, padding="max_length", truncation=True, max_length=160, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for j, i in enumerate(idx):
        a0, n = data["a_rel_start"][i], data["n_answer"][i]
        labels[j, :a0] = -100
        labels[j, a0 + n:] = -100
    with torch.no_grad():
        lg = model(input_ids=enc["input_ids"].to("cuda"),
                   attention_mask=enc["attention_mask"].to("cuda")).logits.float()
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), labels[:, 1:].to("cuda").reshape(-1))
    kept = int((labels != -100).sum())
    print(f"{tag}: CE={ce.item():.4f}  kept_labels={kept}")
    for j, i in enumerate(idx[:2]):
        a0, n = data["a_rel_start"][i], data["n_answer"][i]
        print("   ids:", enc["input_ids"][j, a0 - 1:a0 + n + 2].tolist(),
              "labels:", labels[j, a0 - 1:a0 + n + 2].tolist())
