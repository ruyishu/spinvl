import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
B = "/data/ibu-bi-ai-share-disk"
sys.path.insert(0, f"{D}/scripts")
sys.path.insert(0, f"{D}/src")
from train_final import collate

tok = AutoTokenizer.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", trust_remote_code=True)
data = torch.load(f"{D}/data/geo_final.pt", map_location="cpu")
model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True).eval()

idx = [0, 1, 2, 3]
ids, mask, labels, vis, vmask, *_ = collate(idx, data, tok, 160, False)
print("eos_id", tok.eos_token_id, "pad_id", tok.pad_token_id, "eos_str", repr(tok.eos_token))
for j, i in enumerate(idx):
    a0, n = data["a_rel_start"][i], data["n_answer"][i]
    kept = (labels[j] != -100).nonzero().flatten().tolist()
    print(f"i={i} a0={a0} n={n} kept={kept} ans={data['answers'][i]!r}")

with torch.no_grad():
    lg = model(input_ids=ids.to("cuda"), attention_mask=mask.to("cuda")).logits.float()
ce = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), labels[:, 1:].to("cuda").reshape(-1))
print("CE", round(ce.item(), 4), "kept_total", int((labels != -100).sum()))
