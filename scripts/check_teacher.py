"""验证：Qwen3-VL-4B 老师能加载、能出 caption；师生词表是否逐 id 对齐。"""

import json
import os
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

B = "/data/ibu-bi-ai-share-disk"
TEACHER = f"{B}/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots"
STUDENT = f"{B}/zk/model/Qwen3-1.7B"
IMG_DIR = "/home/powerop/coco_data/val2017"


def find_snapshot(path):
    subs = [os.path.join(path, s) for s in os.listdir(path)]
    subs = [s for s in subs if os.path.isdir(s)]
    return sorted(subs)[-1]


tpath = find_snapshot(TEACHER)
print("teacher snapshot:", tpath)
t0 = time.time()
proc = AutoProcessor.from_pretrained(tpath, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    tpath, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
).eval()
print(f"loaded in {time.time()-t0:.1f}s")

imgs = sorted(os.listdir(IMG_DIR))[:3]
for name in imgs:
    im = Image.open(os.path.join(IMG_DIR, name)).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one detailed sentence."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[im], return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    print(name, "->", proc.batch_decode(gen, skip_special_tokens=True)[0].strip())

# 词表对齐
t_tok = AutoTokenizer.from_pretrained(tpath, trust_remote_code=True)
s_tok = AutoTokenizer.from_pretrained(STUDENT, trust_remote_code=True)
print("teacher vocab", len(t_tok), "student vocab", len(s_tok))
n = min(len(t_tok), len(s_tok))
mismatch = sum(1 for i in range(n) if t_tok.convert_ids_to_tokens(i) != s_tok.convert_ids_to_tokens(i))
print("id->token mismatch:", mismatch, "/", n)
print("MODEL_OK")
