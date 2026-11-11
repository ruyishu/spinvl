"""构建全量去重 + 选项顺序随机重排 (1:1:1:1) + Yes/No 严格 1:1 + 自然图像四选一 (9,000) 的混合数据集。

严格满足全部审阅要求：
1. 6 榜全量评测图指纹库（MMStar 1500 + Hallusion 951 + ChartQA 2500 + OCRBench 1000 + MathVista 1000 + Geometry3K 601 = 7,552 张），
   强制 assert len(bench_dhashes) == 7552；
2. 图像去重（SHA1 精确匹配 或 64-bit dHash 汉明距离 <= 2）同时覆盖复用的旧数据（mix + open_supp）与新抽取的 nat_supp；
   由于 LazyFeats 通过 (voff[i], vlen[i]) 随机寻址 .feats.bin，剔除旧数据中的重复样本只需过滤 meta.pt 下标，无需重跑 8B ViT；
3. A/B/C/D 选项重排统计成功/失败数，解析失败或非四选一的异常条目显式过滤，尾数按 4 取整并强制 assert A == B == C == D；
4. Yes/No 题干通过正则安全剥离前后缀并强制 assert 格式合规；全部使用真实人工标注的 Yes/No（不合成伪造题），
   结合旧 mix 与新抽的 vstar 判断题，强制 assert 全局 Yes == No == 2600（共 5,200 条，严格 1:1）；
5. 新抽自然图像四选一统一为 9,000 条（A/B/C/D 各 2,250 条，严格 1:1:1:1），支持 --part 0/1/2 三卡并行抽取 8B 视觉特征。
"""

import argparse
import gc
import glob
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402
import build_open_supp as bos  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
D = f"{B}/zk/project/rose-distill"
TEACHER = f"{B}/zk/model/Qwen3-VL-8B-Instruct"

COLORS = [
    "white", "black", "red", "blue", "green", "yellow", "brown",
    "gray", "grey", "orange", "pink", "purple", "tan", "silver", "gold",
]
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b", re.I)


def img_dhash(img: Image.Image) -> int:
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def build_bench_fingerprints(cache_path=f"{D}/data/bench_fingerprints_7552.pt"):
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, map_location="cpu", weights_only=False)
        sha_set, dhashes = obj["sha_set"], obj["dhashes"]
        assert len(dhashes) == 7552, f"Cached fingerprint count mismatch: {len(dhashes)} != 7552"
        print(f"Loaded cached bench fingerprints: {len(dhashes)} images (SHA1 unique={len(sha_set)})", flush=True)
        return sha_set, dhashes

    sha_set, dhash_list = set(), []
    per_bench = {}
    for bname in bd.BENCHES:
        items = bd.load_bench(bname)
        cnt = 0
        for it in items:
            raw = it.get("img_bytes")
            if raw is None and it.get("img_path"):
                raw = open(it["img_path"], "rb").read()
            assert raw is not None and len(raw) > 0, f"Empty image in bench {bname} item {it.get('id')}"
            sha_set.add(hashlib.sha1(raw).digest())
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            dhash_list.append(img_dhash(im))
            cnt += 1
        per_bench[bname] = cnt
    dhashes = np.array(dhash_list, dtype=np.uint64)
    print(f"Built bench fingerprints: per_bench={per_bench}, total={len(dhashes)}", flush=True)
    assert len(dhashes) == 7552, f"Expected 7552 benchmark images across 6 benches, got {len(dhashes)}"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"sha_set": sha_set, "dhashes": dhashes, "per_bench": per_bench}, cache_path)
    return sha_set, dhashes


def is_dup_hash(sha_digest: bytes, h_val: int, sha_set: set, bench_dhashes: np.ndarray) -> bool:
    if sha_digest and sha_digest in sha_set:
        return True
    xor = np.bitwise_xor(bench_dhashes, np.uint64(h_val))
    u8 = xor.view(np.uint8).reshape(-1, 8)
    pop = np.unpackbits(u8, axis=1).sum(axis=1)
    return bool(np.any(pop <= 2))


def is_dup_image(raw_bytes: bytes, img: Image.Image, sha_set: set, bench_dhashes: np.ndarray) -> bool:
    d = hashlib.sha1(raw_bytes).digest() if raw_bytes else b""
    return is_dup_hash(d, img_dhash(img), sha_set, bench_dhashes)


def _hash_one_file(p: str):
    if not os.path.exists(p):
        return p, None
    raw_b = open(p, "rb").read()
    im = Image.open(io.BytesIO(raw_b)).convert("RGB")
    return p, (hashlib.sha1(raw_b).digest(), img_dhash(im))


def get_existing_img_hashes(paths: list, cache_path=f"{D}/data/existing_img_hashes_v2.pt"):
    cached = torch.load(cache_path, map_location="cpu", weights_only=False) if os.path.exists(cache_path) else {}
    todo = [p for p in paths if p not in cached]
    if todo:
        print(f"Hashing {len(todo)} existing images with 24 threads...", flush=True)
        with ThreadPoolExecutor(max_workers=24) as ex:
            for p, res in ex.map(_hash_one_file, todo):
                if res is not None:
                    cached[p] = res
        torch.save(cached, cache_path)
    return cached


def manifest_image_paths(out_dir: str, ids: list):
    """优先使用 manifest 的 image 字段；只有没有 manifest 记录时才回退到 img/{id}.jpg。"""
    man = os.path.join(out_dir, "manifest.jsonl")
    mapping = {}
    if os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("id") is not None and r.get("image"):
                    mapping[str(r["id"])] = os.path.realpath(os.path.join(out_dir, r["image"]))
    return {str(sid): mapping.get(str(sid), os.path.realpath(f"{out_dir}/img/{sid}.jpg")) for sid in ids}


def strip_qa_wrapper(prompt: str) -> str:
    body = re.sub(r"^\s*Question:\s*", "", prompt, flags=re.S)
    body = re.sub(r"\s*Answer:\s*$", "", body, flags=re.S)
    return body.strip()


MC_SPLIT_RE = re.compile(
    r"^(.*?)(?:\n|\s*)(?<![A-Za-z0-9])A\.\s*(.*?)\nB\.\s*(.*?)\nC\.\s*(.*?)\nD\.\s*(.*?)(?:\nE\.\s*(.*))?$",
    re.S,
)


def permute_mc_prompt(raw_prompt: str, gold_letter: str, target_idx: int, rng: random.Random):
    """解析题目中的 A/B/C/D 选项并重排到 target_idx (0..3)。返回 (new_prompt, new_letter, ok)。"""
    orig_idx = "ABCD".find(gold_letter.strip().upper())
    if orig_idx < 0:
        return raw_prompt, gold_letter, False
    body = strip_qa_wrapper(raw_prompt)
    m = MC_SPLIT_RE.match(body)
    if not m:
        return raw_prompt, gold_letter, False
    stem = m.group(1).strip()
    opts = [m.group(2).strip(), m.group(3).strip(), m.group(4).strip(), m.group(5).strip()]
    if not stem or any(len(o) == 0 for o in opts) or len(set(opts)) < 4:
        return raw_prompt, gold_letter, False
    gold_text = opts[orig_idx]
    others = [opts[i] for i in range(4) if i != orig_idx]
    rng.shuffle(others)
    new_opts = others[:target_idx] + [gold_text] + others[target_idx:]
    new_letter = "ABCD"[target_idx]
    new_prompt = bd.mc_prompt(stem, new_opts)
    assert new_prompt.startswith("Question: ") and new_prompt.endswith("\nAnswer:")
    return new_prompt, new_letter, True


def format_yn_prompt(raw_prompt_or_q: str, ans_str: str):
    """安全格式化 Yes/No 题干并断言前后缀。"""
    label = "Yes" if ans_str.strip().lower().startswith("yes") else "No"
    q_body = strip_qa_wrapper(raw_prompt_or_q)
    q_body = re.sub(r"\s*Please answer Yes or No\.?\s*$", "", q_body, flags=re.I).strip()
    new_prompt = bd.yn_prompt(q_body)
    assert new_prompt.startswith("Question: ") and new_prompt.endswith("Please answer Yes or No.\nAnswer:")
    return new_prompt, label


def filter_and_rebalance_existing(tok, sha_set, bench_dhashes, write_out=True):
    """对已有的 mix (24,082) 与 open_supp (8,334) 做全量 6 榜图像去重 + 选项顺序 1:1:1:1 重排 + Yes/No 规整。"""
    rng = random.Random(42)
    stats = {
        "mix_total": 0, "mix_dup_dropped": 0, "mix_mc_ok": 0, "mix_mc_fail": 0,
        "mix_seq_dropped": 0, "open_total": 0, "open_dup_dropped": 0,
    }
    mc_records = []
    yn_counts = Counter()
    mc_counts = Counter()

    mix_shards = sorted(glob.glob(f"{D}/data/mix/shard*.meta.pt"))
    open_shards = sorted(glob.glob(f"{D}/data/open_supp/shard*.meta.pt"))
    loaded_mix = [(mp, torch.load(mp, map_location="cpu")) for mp in mix_shards]
    loaded_open = [(mp, torch.load(mp, map_location="cpu")) for mp in open_shards]
    mix_path_map = manifest_image_paths(f"{D}/data/mix", [sid for _, m in loaded_mix for sid in m["ids"]])
    open_path_map = manifest_image_paths(f"{D}/data/open_supp", [sid for _, m in loaded_open for sid in m["ids"]])
    all_img_paths = list(mix_path_map.values()) + list(open_path_map.values())
    img_hashes = get_existing_img_hashes(all_img_paths)
    bad_sha, bad_dhash = set(), set()
    for p, h_pair in img_hashes.items():
        if h_pair is not None and is_dup_hash(h_pair[0], h_pair[1], sha_set, bench_dhashes):
            bad_sha.add(h_pair[0])
            bad_dhash.add(h_pair[1])

    def bad_image(h_pair):
        return h_pair is None or h_pair[0] in bad_sha or h_pair[1] in bad_dhash

    filtered_shards = []
    for s_idx, (mp, m) in enumerate(loaded_mix):
        keep_indices = []
        for i, sid in enumerate(m["ids"]):
            stats["mix_total"] += 1
            img_p = mix_path_map[str(sid)]
            h_pair = img_hashes.get(img_p)
            if bad_image(h_pair):
                stats["mix_dup_dropped"] += 1
                continue
            ans = m["answers"][i].strip()
            pr = m["prompts"][i]
            if ans in ("A", "B", "C", "D", "E"):
                pr_test, ans_test, ok = permute_mc_prompt(pr, ans, 0, rng)
                if not ok:
                    stats["mix_mc_fail"] += 1
                    continue
                pids = tok(pr_test + " " + ans_test)["input_ids"]
                aids = tok(" " + ans_test)["input_ids"]
                if len(pids) > 220 or len(aids) == 0 or pids[-len(aids):] != aids:
                    stats["mix_seq_dropped"] += 1
                    continue
                stats["mix_mc_ok"] += 1
                mc_records.append((s_idx, len(keep_indices)))
            else:
                if ans.lower() in ("yes", "no"):
                    pr_test, ans_test = format_yn_prompt(pr, ans)
                else:
                    pr_test, ans_test = bd.qa_prompt(strip_qa_wrapper(pr)), ans
                pids = tok(pr_test + " " + ans_test)["input_ids"]
                aids = tok(" " + ans_test)["input_ids"]
                if len(pids) > 220 or len(aids) == 0 or pids[-len(aids):] != aids:
                    stats["mix_seq_dropped"] += 1
                    continue
            keep_indices.append(i)
        filtered_shards.append((mp, m, keep_indices))

    rem = len(mc_records) % 4
    drop_mc_set = set(mc_records[-rem:]) if rem > 0 else set()

    out_mix_dir = f"{D}/data/mix_v2"
    if write_out:
        os.makedirs(out_mix_dir, exist_ok=True)
    mc_seq = 0
    kept_mix_total = 0
    mix_man_rows = []

    for s_idx, (mp, m, keep_indices) in enumerate(filtered_shards):
        new_m = {
            "ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [],
            "topk_ids": [], "topk_vals": [], "hidden": [[] for _ in range(4)],
            "voff": [], "vlen": [], "vdim": m["vdim"], "feat_bin": m["feat_bin"], "img_sha1": [],
        }
        for local_k, i in enumerate(keep_indices):
            if (s_idx, local_k) in drop_mc_set:
                continue
            sid = m["ids"][i]
            img_p = mix_path_map[str(sid)]
            sha_hex = img_hashes[img_p][0].hex()
            pr, ans = m["prompts"][i], m["answers"][i].strip()
            if ans in ("A", "B", "C", "D"):
                t_idx = mc_seq % 4
                mc_seq += 1
                pr_new, ans_new, ok = permute_mc_prompt(pr, ans, t_idx, rng)
                assert ok
                mc_counts[ans_new] += 1
            elif ans.lower() in ("yes", "no"):
                pr_new, ans_new = format_yn_prompt(pr, ans)
                yn_counts[ans_new] += 1
            else:
                pr_new, ans_new = bd.qa_prompt(strip_qa_wrapper(pr)), ans
            pids = tok(pr_new + " " + ans_new)["input_ids"]
            aids = tok(" " + ans_new)["input_ids"]
            new_m["ids"].append(sid)
            new_m["prompts"].append(pr_new)
            new_m["answers"].append(ans_new)
            new_m["a_rel_start"].append(len(pids) - len(aids))
            new_m["n_answer"].append(len(aids))
            new_m["topk_ids"].append(None)
            new_m["topk_vals"].append(None)
            for li in range(4):
                new_m["hidden"][li].append(None)
            new_m["voff"].append(m["voff"][i])
            new_m["vlen"].append(m["vlen"][i])
            new_m["img_sha1"].append(sha_hex)
            rel_img = os.path.relpath(img_p, f"{D}/data/mix").replace("\\", "/")
            mix_man_rows.append({"id": sid, "prompt": pr_new, "answer": ans_new, "image": rel_img, "img_sha1": sha_hex})
            kept_mix_total += 1
        if write_out:
            torch.save(new_m, os.path.join(out_mix_dir, os.path.basename(mp)))
    if write_out:
        with open(os.path.join(out_mix_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
            for r in mix_man_rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")

    assert len(set(mc_counts[l] for l in "ABCD")) == 1, f"mix_v2 MC 1:1:1:1 failed: {mc_counts}"
    print(f"[mix_v2] kept={kept_mix_total}, stats={stats}, MC={dict(mc_counts)}, YN={dict(yn_counts)}", flush=True)

    out_open_dir = f"{D}/data/open_supp_v2"
    if write_out:
        os.makedirs(out_open_dir, exist_ok=True)
    kept_open_total = 0
    open_man_rows = []
    for mp, m in loaded_open:
        new_m = {
            "ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [],
            "topk_ids": [], "topk_vals": [], "hidden": [[] for _ in range(4)],
            "voff": [], "vlen": [], "vdim": m["vdim"], "feat_bin": m["feat_bin"], "img_sha1": [],
        }
        for i, sid in enumerate(m["ids"]):
            stats["open_total"] += 1
            img_p = open_path_map[str(sid)]
            h_pair = img_hashes.get(img_p)
            if bad_image(h_pair):
                stats["open_dup_dropped"] += 1
                continue
            sha_hex = h_pair[0].hex()
            ans = m["answers"][i].strip()
            if ans.lower() in ("yes", "no"):
                continue
            pr = m["prompts"][i]
            pids = tok(pr + " " + ans)["input_ids"]
            aids = tok(" " + ans)["input_ids"]
            if len(aids) == 0 or pids[-len(aids):] != aids:
                stats["open_seq_dropped"] = stats.get("open_seq_dropped", 0) + 1
                continue
            new_m["ids"].append(sid)
            new_m["prompts"].append(pr)
            new_m["answers"].append(ans)
            new_m["a_rel_start"].append(len(pids) - len(aids))
            new_m["n_answer"].append(len(aids))
            new_m["topk_ids"].append(None)
            new_m["topk_vals"].append(None)
            for li in range(4):
                new_m["hidden"][li].append(None)
            new_m["voff"].append(m["voff"][i])
            new_m["vlen"].append(m["vlen"][i])
            new_m["img_sha1"].append(sha_hex)
            rel_img = os.path.relpath(img_p, f"{D}/data/open_supp").replace("\\", "/")
            open_man_rows.append({"id": sid, "prompt": pr, "answer": ans, "image": rel_img, "img_sha1": sha_hex})
            kept_open_total += 1
        if write_out:
            torch.save(new_m, os.path.join(out_open_dir, os.path.basename(mp)))
    if write_out:
        with open(os.path.join(out_open_dir, "manifest.jsonl"), "w", encoding="utf-8") as mf:
            for r in open_man_rows:
                mf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[open_supp_v2] kept={kept_open_total}, dup_dropped={stats['open_dup_dropped']}", flush=True)
    return yn_counts, mc_counts, stats


def _make_vstar_distractors(q: str, a: str, bucket_answers: list, rng: random.Random):
    """根据 vstar 自然图像问答的题型（方位/颜色/实体）生成 3 个无歧义干扰项。"""
    al = a.lower()
    if "left" in al or "right" in al:
        opp = re.sub(r"\bleft\b", "__TMP__", a, flags=re.I)
        opp = re.sub(r"\bright\b", "left", opp, flags=re.I).replace("__TMP__", "right")
        d2 = re.sub(r"\b(left|right)\b", "top", a, flags=re.I)
        d3 = re.sub(r"\b(left|right)\b", "bottom", a, flags=re.I)
        cands = [opp, d2, d3]
        if len({a, *cands}) == 4:
            return cands
    cm = COLOR_RE.search(a)
    if cm:
        orig_c = cm.group(1).lower()
        other_cs = [c for c in COLORS if c != orig_c and c not in ("grey" if orig_c == "gray" else "gray")]
        rng.shuffle(other_cs)
        cands = [COLOR_RE.sub(c, a, count=1) for c in other_cs[:3]]
        if len({a, *cands}) == 4:
            return cands
    pool = [x for x in bucket_answers if x != a]
    if len(pool) >= 10:
        picked = []
        for _ in range(30):
            cand = rng.choice(pool)
            if cand not in picked and cand != a:
                picked.append(cand)
            if len(picked) == 3:
                return picked
    return None


def build_nat_supp_manifest(tok, sha_set, bench_dhashes, existing_yn: Counter, existing_mc: Counter,
                            out_dir=f"{D}/data/nat_supp", target_yn_each=2600, target_nat_mc=9000,
                            write_images=True):
    """构建自然图像补丁包清单：
    1) 100% 使用真实标注的 Yes/No，将全局 Yes/No 精确配平为 2,600 : 2,600（共 5,200 条，严格 1:1）；
    2) 构建 9,000 条自然图像四选一（A/B/C/D 各 2,250 条，严格 1:1:1:1）。
    """
    assert target_nat_mc % 4 == 0
    rng = random.Random(2026)

    need_yes = target_yn_each - existing_yn["Yes"]
    need_no = target_yn_each - existing_yn["No"]
    assert need_yes > 0 and need_no > 0, f"Invalid YN needs: {need_yes}, {need_no}"
    per_let = target_nat_mc // 4
    print(f"[nat_supp plan] need_yes={need_yes}, need_no={need_no}, need_mc={per_let}x4={target_nat_mc}", flush=True)

    de_dir = glob.glob(f"{B}/.cache/huggingface/hub/datasets--ChenShawn--DeepEyes-Datasets-47k/snapshots/*")[0]
    pf = pq.ParquetFile(f"{de_dir}/data_0.1.2_visual_toolbox_v2.parquet")

    yes_pool, no_pool, raw_other = [], [], []
    bucket_ans = defaultdict(list)
    dup_vstar = 0

    for rg in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(rg, columns=["images", "reward_model", "extra_info"]).to_pylist():
            q = (r.get("extra_info") or {}).get("question", "").strip()
            a = str((r.get("reward_model") or {}).get("ground_truth", "")).strip()
            imgs = r.get("images") or []
            if not q or not a or not imgs:
                continue
            raw_b = imgs[0]["bytes"]
            w0 = a.split(",")[0].split()[0].rstrip(".,").lower()
            if w0 in ("yes", "no"):
                if w0 == "yes" and len(yes_pool) >= need_yes + 400:
                    continue
                if w0 == "no" and len(no_pool) >= need_no + 400:
                    continue
                im = Image.open(io.BytesIO(raw_b)).convert("RGB")
                if is_dup_image(raw_b, im, sha_set, bench_dhashes):
                    dup_vstar += 1
                    continue
                pr_yn, lab_yn = format_yn_prompt(q, w0)
                pids = tok(pr_yn + " " + lab_yn)["input_ids"]
                aids = tok(" " + lab_yn)["input_ids"]
                if len(pids) <= 220 and pids[-len(aids):] == aids:
                    if lab_yn == "Yes":
                        yes_pool.append((q, pr_yn, "Yes", raw_b))
                    else:
                        no_pool.append((q, pr_yn, "No", raw_b))
            else:
                if len(raw_other) >= target_nat_mc + 1500:
                    continue
                im = Image.open(io.BytesIO(raw_b)).convert("RGB")
                if is_dup_image(raw_b, im, sha_set, bench_dhashes):
                    dup_vstar += 1
                    continue
                bkey = " ".join(q.lower().split()[:2])
                bucket_ans[bkey].append(a)
                raw_other.append((q, a, bkey, raw_b))

    mc_pool = []
    for q, a, bkey, raw_b in raw_other:
        dists = _make_vstar_distractors(q, a, bucket_ans[bkey], rng)
        if dists is not None:
            mc_pool.append((q, a, dists, raw_b))

    print(f"[vstar scan] yes_pool={len(yes_pool)}, no_pool={len(no_pool)}, mc_pool={len(mc_pool)}, dup_dropped={dup_vstar}", flush=True)
    assert len(yes_pool) >= need_yes, f"yes_pool {len(yes_pool)} < need_yes {need_yes}"
    assert len(no_pool) >= need_no, f"no_pool {len(no_pool)} < need_no {need_no}"
    assert len(mc_pool) >= target_nat_mc, f"mc_pool {len(mc_pool)} < target_nat_mc {target_nat_mc}"

    rng.shuffle(yes_pool)
    rng.shuffle(no_pool)
    rng.shuffle(mc_pool)

    selected = []
    for idx, (q, pr_yn, lab_yn, raw_b) in enumerate(yes_pool[:need_yes]):
        selected.append({"id": f"nat_yes_{idx}", "prompt": pr_yn, "answer": "Yes", "raw_bytes": raw_b})
    for idx, (q, pr_yn, lab_yn, raw_b) in enumerate(no_pool[:need_no]):
        selected.append({"id": f"nat_no_{idx}", "prompt": pr_yn, "answer": "No", "raw_bytes": raw_b})

    mc_cursor = 0
    for let_idx, let in enumerate("ABCD"):
        kept_let = 0
        while kept_let < per_let:
            stem, gold_opt, dist_opts, raw_b = mc_pool[mc_cursor]
            mc_cursor += 1
            others = list(dist_opts)
            rng.shuffle(others)
            opts = others[:let_idx] + [gold_opt] + others[let_idx:]
            pr_mc = bd.mc_prompt(stem, opts)
            pids = tok(pr_mc + " " + let)["input_ids"]
            aids = tok(" " + let)["input_ids"]
            if len(pids) <= 220 and pids[-len(aids):] == aids:
                selected.append({"id": f"nat_mc_{let}_{kept_let}", "prompt": pr_mc, "answer": let, "raw_bytes": raw_b})
                kept_let += 1

    rng.shuffle(selected)

    final_yn = Counter(existing_yn)
    final_mc = Counter(existing_mc)
    nat_mc_c = Counter()
    for r in selected:
        if r["answer"] in ("Yes", "No"):
            final_yn[r["answer"]] += 1
        elif r["answer"] in ("A", "B", "C", "D"):
            final_mc[r["answer"]] += 1
            nat_mc_c[r["answer"]] += 1

    print(f"NAT_SUPP MC={dict(nat_mc_c)}, GLOBAL YN={dict(final_yn)}, GLOBAL MC={dict(final_mc)}", flush=True)
    assert final_yn["Yes"] == target_yn_each and final_yn["No"] == target_yn_each, f"YN balance failed: {final_yn}"
    assert all(nat_mc_c[l] == per_let for l in "ABCD"), f"nat_supp MC 2250x4 failed: {nat_mc_c}"
    assert len(set(final_mc[l] for l in "ABCD")) == 1, f"Global MC 1:1:1:1 balance failed: {final_mc}"

    if not write_images:
        return selected

    os.makedirs(f"{out_dir}/img", exist_ok=True)
    man_path = f"{out_dir}/manifest.jsonl"
    manifest_rows = []
    with open(man_path, "w") as f:
        for r in selected:
            rid = r["id"]
            rel_p = f"img/{rid}.jpg"
            full_p = f"{out_dir}/{rel_p}"
            if not os.path.exists(full_p):
                im = bos.fit(Image.open(io.BytesIO(r["raw_bytes"])).convert("RGB"))
                im.save(full_p, quality=92)
            row = {"id": rid, "prompt": r["prompt"], "answer": r["answer"], "image": rel_p}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest_rows.append(row)
    print(f"Saved {len(manifest_rows)} rows to {man_path}", flush=True)
    return manifest_rows


def extract_nat_supp_shard(out_dir, part: int, num_parts: int = 3, batch: int = 24):
    man_path = f"{out_dir}/manifest.jsonl"
    rows = [json.loads(l) for l in open(man_path)]
    n = len(rows)
    per = (n + num_parts - 1) // num_parts
    sub_rows = rows[part * per : (part + 1) * per]
    meta_path = f"{out_dir}/shard{part}.meta.pt"
    bin_path = f"{out_dir}/shard{part}.feats.bin"
    if os.path.exists(meta_path):
        print(f"shard {part} already exists at {meta_path}, skip", flush=True)
        return

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

    out = {
        "ids": [], "prompts": [], "answers": [], "a_rel_start": [], "n_answer": [],
        "topk_ids": [], "topk_vals": [], "hidden": [[], [], [], []],
        "voff": [], "vlen": [], "vdim": vdim, "feat_bin": bin_path,
    }
    n_tok, t0 = 0, time.time()
    with open(bin_path, "wb") as fbin:
        for i in range(0, len(sub_rows), batch):
            chunk = sub_rows[i : i + batch]
            imgs, keep, meta_info = [], [] , []
            for r in chunk:
                pr = r["prompt"]
                pt = pr + " " + r["answer"]
                pids = tok(pt)["input_ids"]
                aids = tok(" " + r["answer"])["input_ids"]
                assert len(pids) <= 220 and len(aids) > 0 and pids[-len(aids):] == aids, f"Tokenization mismatch on {r['id']}"
                imgs.append(Image.open(f"{out_dir}/{r['image']}").convert("RGB"))
                keep.append(r)
                meta_info.append((pr, len(pids) - len(aids), len(aids)))
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
            if (i // batch) % 20 == 0:
                print(f"[shard {part}] {len(out['ids'])}/{len(sub_rows)} in {time.time()-t0:.1f}s", flush=True)
    torch.save(out, meta_path)
    print(f"NAT_SUPP_SHARD_DONE {part}: {len(out['ids'])} samples, {n_tok} vtoks in {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["check", "prepare", "extract"], default="check")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--num-parts", type=int, default=3)
    ap.add_argument("--out-dir", default=f"{D}/data/nat_supp")
    args = ap.parse_args()

    if args.step in ("check", "prepare"):
        write = (args.step == "prepare")
        sha_set, bench_dhashes = build_bench_fingerprints()
        tok = AutoProcessor.from_pretrained(TEACHER, trust_remote_code=True).tokenizer
        yn_counts, mc_counts, _ = filter_and_rebalance_existing(tok, sha_set, bench_dhashes, write_out=write)
        build_nat_supp_manifest(tok, sha_set, bench_dhashes, yn_counts, mc_counts,
                                out_dir=args.out_dir, write_images=write)
        print(f"STEP_{args.step.upper()}_ALL_DONE", flush=True)
    elif args.step == "extract":
        extract_nat_supp_shard(args.out_dir, args.part, args.num_parts)


if __name__ == "__main__":
    main()
