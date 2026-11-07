import torch
from transformers import AutoTokenizer

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
tok = AutoTokenizer.from_pretrained("/data/ibu-bi-ai-share-disk/zk/model/Qwen3-1.7B", trust_remote_code=True)
for tag, path in (("4B", f"{D}/data/geo_final.pt"), ("8B", f"{D}/data/geo_8b.pt")):
    d = torch.load(path, map_location="cpu")
    hit = tot = 0
    for i, a in enumerate(d["answers"]):
        gold = tok(" " + a)["input_ids"]
        n = min(len(gold), d["topk_ids"][i].shape[0])
        hit += sum(int(d["topk_ids"][i][k][0].item() == gold[k]) for k in range(n))
        tot += n
    print(f"{tag} teacher_top1_hit={hit/max(tot,1):.3f}  n={len(d['ids'])}  vdim={d['vfeat'][0].shape[-1]}")
