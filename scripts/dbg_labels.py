import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
sys.path.insert(0, "/data/ibu-bi-ai-share-disk/zk/project/rose-distill/src")
sys.path.insert(0, "/data/ibu-bi-ai-share-disk/zk/project/rose-distill/scripts")
from train_final import collate

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
tok = AutoTokenizer.from_pretrained("/data/ibu-bi-ai-share-disk/zk/model/Qwen3-1.7B", trust_remote_code=True)
print("eos_token:", repr(tok.eos_token), "eos_id:", tok.eos_token_id)
data = torch.load(f"{D}/data/geo_final.pt", map_location="cpu")
idx = [0, 1, 2]
ids, mask, labels, vis, vmask, *_ = collate(idx, data, tok, 160, False)
for j, i in enumerate(idx):
    a0, n = data["a_rel_start"][i], data["n_answer"][i]
    print(f"\nsample {i} a0={a0} n={n} eos={tok.eos_token_id}")
    print(" ids  :", ids[j, max(0, a0 - 3) : a0 + n + 3].tolist())
    print(" label:", labels[j, max(0, a0 - 3) : a0 + n + 3].tolist())
    print(" text tail:", repr(tok.decode(ids[j, max(0, a0 - 3) : a0 + n + 3])))
    print(" n_label_kept:", int((labels[j] != -100).sum()))
