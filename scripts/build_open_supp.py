"""从共享盘 Visual-CoT 已缓存分卷 (cot_images_00..02) 中提取开放短答案训练数据（6500 条 docvqa 图表/表格/OCR + 2500 条 gqa），
复用已解出图片，捕获分卷末尾 ReadError，并仅加载 Qwen3-VL-8B 视觉塔写出 memmap 分片（shard0..3.meta.pt）。
"""

import argparse
import gc
import glob
import io
import json
import os
import random
import re
import tarfile
import time

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

B = "/data/ibu-bi-ai-share-disk"
D = f"{B}/zk/project/rose-distill"
TEACHER = f"{B}/zk/model/Qwen3-VL-8B-Instruct"
PROMPT_TMPL = "Question: {q}\nAnswer:"
BBOX_PAT = re.compile(r"\s*Please provide the bounding box coordinate.*$", re.I | re.S)


class ConcatSeekFile:
    def __init__(self, paths):
        self.files = [open(p, "rb") for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.cum = [0]
        for s in self.sizes:
            self.cum.append(self.cum[-1] + s)
        self.total = self.cum[-1]
        self.pos = 0

    def tell(self):
        return self.pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self.pos = max(0, min(self.total, offset))
        elif whence == os.SEEK_CUR:
            self.pos = max(0, min(self.total, self.pos + offset))
        elif whence == os.SEEK_END:
            self.pos = max(0, min(self.total, self.total + offset))
        return self.pos

    def read(self, n=-1):
        if self.pos >= self.total:
            return b""
        if n < 0 or self.pos + n > self.total:
            n = self.total - self.pos
        out = bytearray()
        while len(out) < n and self.pos < self.total:
            idx = 0
            while idx + 1 < len(self.cum) and self.cum[idx + 1] <= self.pos:
                idx += 1
            f = self.files[idx]
            local_pos = self.pos - self.cum[idx]
            avail = self.sizes[idx] - local_pos
            take = min(n - len(out), avail)
            f.seek(local_pos)
            chunk = f.read(take)
            if not chunk:
                break
            out.extend(chunk)
            self.pos += len(chunk)
        return bytes(out)

    def close(self):
        for f in self.files:
            f.close()


def fit(img, max_pixels=392 * 392):
    w, h = img.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        w, h = int(w * s), int(h * s)
    w, h = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
    return img.resize((w, h), Image.BICUBIC)


def clean_question(raw_q):
    q = raw_q.replace("<image>", "").strip()
    q = BBOX_PAT.sub("", q).strip()
    return q


def build_manifest(out_dir, max_ans_chars=24):
    os.makedirs(f"{out_dir}/img", exist_ok=True)
    man_path = f"{out_dir}/manifest.jsonl"
    if os.path.exists(man_path) and sum(1 for _ in open(man_path)) >= 3000:
        print(f"manifest exists at {man_path}, skip unpack", flush=True)
        return [json.loads(l) for l in open(man_path)]

    vc = glob.glob(f"{B}/.cache/huggingface/hub/datasets--deepcs233--Visual-CoT/snapshots/*")[0]
    raw_data = json.load(open(f"{vc}/viscot_363k.json"))
    rng = random.Random(42)
    quotas = {"docvqa": 6500, "gqa": 2500}
    buckets = {k: [] for k in quotas}
    for r in raw_data:
        ds = r.get("dataset")
        if ds not in buckets:
            continue
        convs = r.get("conversations") or []
        if len(convs) < 2:
            continue
        q = clean_question(convs[0].get("value", ""))
        a = str(convs[-1].get("value", "")).strip().rstrip(".")
        imgs = r.get("image") or []
        if not q or not a or not imgs or len(a) > max_ans_chars or "\n" in a:
            continue
        tar_rel = imgs[0].replace("cot/", "cot_image_data/", 1)
        buckets[ds].append((q, a, tar_rel))

    picked = []
    for ds, items in buckets.items():
        rng.shuffle(items)
        for idx, (q, a, tar_rel) in enumerate(items[: quotas[ds]]):
            picked.append({"id": f"{ds}_{idx}", "question": q, "answer": a, "tar_rel": tar_rel})

    saved_map = {}
    wanted = {}
    for r in picked:
        rel_p = f"img/{r['id']}.jpg"
        if os.path.exists(f"{out_dir}/{rel_p}"):
            saved_map[r["id"]] = rel_p
        else:
            wanted.setdefault(r["tar_rel"], []).append(r["id"])
    print(f"Picked {len(picked)} QA pairs (already cached={len(saved_map)}, need tar={len(wanted)} unique imgs)", flush=True)

    if wanted:
        parts = sorted(glob.glob(f"{vc}/cot_images_tar_split/cot_images_*"))
        csf = ConcatSeekFile(parts)
        t0 = time.time()
        tf = tarfile.open(fileobj=csf, mode="r:")
        while True:
            try:
                member = tf.next()
            except (tarfile.ReadError, EOFError, OSError) as e:
                print(f"Reached end of cached tar splits ({e}), stopping scan.", flush=True)
                break
            if member is None:
                break
            if member.name in wanted:
                try:
                    raw_bytes = tf.extractfile(member).read()
                    img = fit(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                    rids = wanted.pop(member.name)
                    rel_p = f"img/{rids[0]}.jpg"
                    img.save(f"{out_dir}/{rel_p}", quality=92)
                    for rid in rids:
                        saved_map[rid] = rel_p
                except Exception:
                    pass
                if len(saved_map) % 500 == 0:
                    print(f"  total ready {len(saved_map)}/{len(picked)} items in {time.time()-t0:.1f}s", flush=True)
                if not wanted:
                    break
        csf.close()

    rows = []
    with open(man_path, "w") as f:
        for r in picked:
            p = saved_map.get(r["id"])
            if not p:
                continue
            item = {"id": r["id"], "question": r["question"], "answer": r["answer"], "image": p}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            rows.append(item)
    print(f"MANIFEST DONE: {len(rows)} rows", flush=True)
    return rows


def extract_vision_shards(out_dir, rows, shard_size=2250, batch=24):
    proc = AutoProcessor.from_pretrained(TEACHER, trust_remote_code=True)
    ip = proc.image_processor
    tok = proc.tokenizer
    vdim = AutoConfig.from_pretrained(TEACHER, trust_remote_code=True).text_config.hidden_size
    full = AutoModelForImageTextToText.from_pretrained(TEACHER, dtype=torch.bfloat16, trust_remote_code=True).eval()
    vis = (getattr(full, "visual", None) or full.model.visual).to("cuda:0").eval()
    for p in vis.parameters():
        p.requires_grad_(False)
    merge = getattr(vis.config, "spatial_merge_size", None) or getattr(vis.config, "merge_size", 2)
    del full
    gc.collect()
    torch.cuda.empty_cache()

    n_shards = (len(rows) + shard_size - 1) // shard_size
    for k in range(n_shards):
        meta_path = f"{out_dir}/shard{k}.meta.pt"
        if os.path.exists(meta_path):
            print(f"shard {k} exists, skip", flush=True)
            continue
        bin_path = f"{out_dir}/shard{k}.feats.bin"
        part = rows[k * shard_size : (k + 1) * shard_size]
        out = {"ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [],
               "topk_ids": [], "topk_vals": [], "hidden": [[], [], [], []],
               "voff": [], "vlen": [], "vdim": vdim, "feat_bin": bin_path}
        n_tok, t0 = 0, time.time()
        with open(bin_path, "wb") as fbin:
            for i in range(0, len(part), batch):
                chunk = part[i : i + batch]
                imgs, keep, meta_info = [], [], []
                for r in chunk:
                    pr = PROMPT_TMPL.format(q=r["question"])
                    pt = pr + " " + r["answer"]
                    pids = tok(pt)["input_ids"]
                    aids = tok(" " + r["answer"])["input_ids"]
                    if len(pids) > 240 or len(aids) == 0 or pids[-len(aids):] != aids:
                        continue
                    imgs.append(Image.open(f"{out_dir}/{r['image']}").convert("RGB"))
                    keep.append(r)
                    meta_info.append((pr, len(pids) - len(aids), len(aids)))
                if not keep:
                    continue
                pvs, grids = [], []
                for im in imgs:
                    o = ip(images=[im], return_tensors="pt")
                    pvs.append(o["pixel_values"].to(torch.bfloat16))
                    grids.append(o["image_grid_thw"][0])
                cat_pv = torch.cat(pvs, dim=0).to("cuda:0")
                grid_t = torch.stack(grids).to("cuda:0")
                with torch.no_grad():
                    vo = vis(cat_pv, grid_thw=grid_t)
                vf = (vo[0] if isinstance(vo, (tuple, list)) else vo).to(torch.float16).cpu()
                counts = [int(g[0] * g[1] * g[2] // (merge * merge)) for g in grid_t.tolist()]
                off = 0
                for r, (pr, a0, na), nv in zip(keep, meta_info, counts):
                    feat = vf[off : off + nv]
                    off += nv
                    fbin.write(feat.numpy().tobytes())
                    out["ids"].append(r["id"])
                    out["prompts"].append(pr)
                    out["answers"].append(r["answer"])
                    out["a_rel_start"].append(a0)
                    out["n_answer"].append(na)
                    out["topk_ids"].append(None)
                    out["topk_vals"].append(None)
                    for li in range(4):
                        out["hidden"][li].append(None)
                    out["voff"].append(n_tok)
                    out["vlen"].append(nv)
                    n_tok += nv
        torch.save(out, meta_path)
        print(f"SHARD_DONE {k}: {len(out['ids'])} samples, {n_tok} vtoks in {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=f"{D}/data/open_supp")
    ap.add_argument("--per-ds", type=int, default=3000)
    args = ap.parse_args()
    rows = build_manifest(args.out_dir)
    extract_vision_shards(args.out_dir, rows)
    open(f"{args.out_dir}/open_supp.ok", "w").write("ok\n")
    print("OPEN_SUPP_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
