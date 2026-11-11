"""6 个公开榜的数据加载（统一成 item 字典）+ 提示词构造（对齐 mix 训练时的 chart A. / B. 与 Yes/No 样式）。"""

import base64
import csv
import io
import json

B = "/data/ibu-bi-ai-share-disk"
RAW = B + "/zk/project/bencheval/data/raw"
MV = B + "/zk/project/mm-opsd/data/processed-mv"
BENCHES = ["mmstar", "hallusion", "chartqa", "ocrbench", "mathvista", "geometry3k"]


def qa_prompt(q):
    return f"Question: {q}\nAnswer:"


def mc_prompt(q, choices):
    """对齐 mix 中 chart_* 的 A. xxx / B. xxx 选项格式。"""
    return "Question: " + q + "\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices)) + "\nAnswer:"


def yn_prompt(q):
    """对齐 mix 中 think_* 的 Yes/No 判断题格式。"""
    return f"Question: {q} Please answer Yes or No.\nAnswer:"


def fit(img, max_pixels):
    from PIL import Image
    w, h = img.size
    if w * h > max_pixels:
        s = (max_pixels / (w * h)) ** 0.5
        w, h = int(w * s), int(h * s)
    w, h = max(32, round(w / 32) * 32), max(32, round(h / 32) * 32)
    return img.resize((w, h), Image.BICUBIC)


def get_image(it, max_pixels):
    from PIL import Image
    src = io.BytesIO(it["img_bytes"]) if it.get("img_bytes") is not None else it["img_path"]
    img = Image.open(src).convert("RGB")
    return fit(img, max_pixels) if max_pixels else img


def _parse_opts(opts):
    import re
    opts = opts.strip()
    pos, start = [], 0
    for l in "ABCD":
        if l == "A":
            m = re.search(r"^\s*(?:A\s*[:.]|\(A\))\s*", opts[start:])
        else:
            m = re.search(rf"(?:[,;\n]\s*|\s+)(?:{l}\s*[:.]|\({l}\))\s*", opts[start:])
        if not m:
            break
        i = start + m.start()
        j = start + m.end()
        pos.append((i, j))
        start = j
    return [opts[j : (pos[k + 1][0] if k + 1 < len(pos) else len(opts))].strip().rstrip(",;") for k, (i, j) in enumerate(pos)]


def load_mmstar():
    import re
    csv.field_size_limit(10 ** 9)
    items = []
    with open(RAW + "/MMStar/MMStar.tsv", newline="") as f:
        rd = csv.reader(f, delimiter="\t")
        ix = {h: i for i, h in enumerate(next(rd))}
        for row in rd:
            q, ans = row[ix["question"]], row[ix["answer"]].strip()
            m_opt = re.search(r"(?:\n|\s+)Options:\s*", q)
            m_ch = re.search(r"(?:\n|\s+)Choices:\s*", q)
            m_in = re.search(r"\n\s*\(A\)\s*", q)
            if m_opt:
                stem, opts = q[:m_opt.start()], q[m_opt.end():]
                choices = _parse_opts(opts)
            elif m_ch:
                stem, opts = q[:m_ch.start()], q[m_ch.end():]
                lines = [ln.strip() for ln in opts.strip().splitlines() if ln.strip()]
                if len(lines) >= 2:
                    choices = [re.sub(r"^(?:\([A-Za-z]\)|[A-Za-z][.:])\s*", "", ln) for ln in lines]
                else:
                    choices = _parse_opts(opts)
            elif m_in:
                stem, opts = q[:m_in.start()], q[m_in.start():]
                lines = [ln.strip() for ln in opts.strip().splitlines() if ln.strip()]
                choices = [re.sub(r"^(?:\([A-Za-z]\)|[A-Za-z][.:])\s*", "", ln) for ln in lines]
            else:
                stem, choices = q, _parse_opts(q)
            stem = re.sub(r"^Hint:.*?\nQuestion:\s*", "", stem.strip(), flags=re.S)
            gi = ord(ans[0].upper()) - 65 if ans else -1
            ok = len(choices) >= 2 and 0 <= gi < len(choices)
            assert ok, f"MMStar choice parse failed on index={row[ix['index']]}: ans={ans!r} choices={choices!r} q={q!r}"
            items.append({"id": f"mmstar:{row[ix['index']]}", "prompt": mc_prompt(stem.strip(), choices),
                          "choices": choices, "gold_idx": gi, "golds": [ans],
                          "cat": row[ix["category"]], "img_bytes": base64.b64decode(row[ix["image"]])})
    return items


def _rows(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def load_hallusion():
    items = []
    for i, r in enumerate(_rows(RAW + "/HallusionBench/image-00000-of-00001.parquet")):
        if not r["image"] or not r["image"]["bytes"]:
            continue
        gi = 0 if str(r["gt_answer"]).strip() == "1" else 1
        items.append({"id": f"hallusion:{i}", "prompt": yn_prompt(r["question"]), "choices": ["Yes", "No"],
                      "gold_idx": gi, "golds": ["Yes" if gi == 0 else "No"], "cat": r["category"], "img_bytes": r["image"]["bytes"]})
    return items


def load_chartqa():
    return [{"id": f"chartqa:{i}", "prompt": qa_prompt(r["query"]), "choices": [], "gold_idx": -1, "golds": list(r["label"]),
             "cat": f"human_or_machine={r['human_or_machine']}", "img_bytes": r["image"]["bytes"]}
            for i, r in enumerate(_rows(RAW + "/ChartQA/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet"))]


def load_ocrbench():
    return [{"id": f"ocrbench:{i}", "prompt": qa_prompt(r["question"]), "choices": [], "gold_idx": -1, "golds": list(r["answer"]),
             "cat": r["question_type"], "img_bytes": r["image"]["bytes"]}
            for i, r in enumerate(_rows(RAW + "/OCRBench/test-00000-of-00001.parquet"))]


def _jsonl(name):
    return [json.loads(l) for l in open(f"{MV}/{name}.jsonl")]


def load_mathvista():
    items = []
    for r in _jsonl("mathvista_testmini"):
        ch = list(r.get("choices") or [])
        gi = ch.index(r["answer"]) if r["question_type"] == "multi_choice" and r["answer"] in ch else -1
        items.append({"id": r["id"], "prompt": mc_prompt(r["question"], ch) if ch else qa_prompt(r["question"]),
                      "choices": ch, "gold_idx": gi, "golds": [r["answer"]], "cat": f"{r['question_type']}/{r['answer_type']}",
                      "img_path": f"{MV}/{r['image']}", "answer_type": r["answer_type"], "precision": r.get("precision")})
    return items


def load_geometry3k():
    return [{"id": r["id"], "prompt": qa_prompt(r["question"]), "choices": [], "gold_idx": -1, "golds": [r["answer"]],
             "cat": "geometry3k", "img_path": f"{MV}/{r['image']}"} for r in _jsonl("geometry3k_test")]


LOADERS = {"mmstar": load_mmstar, "hallusion": load_hallusion, "chartqa": load_chartqa, "ocrbench": load_ocrbench,
           "mathvista": load_mathvista, "geometry3k": load_geometry3k}


def load_bench(name, limit=0):
    items = LOADERS[name]()
    return items[:limit] if limit else items


def check_data(limit=0):
    for b in BENCHES:
        items = load_bench(b, limit)
        cats = {}
        for it in items:
            cats[it["cat"]] = cats.get(it["cat"], 0) + 1
        sizes = [get_image(it, 153664).size for it in items[:3]]
        print(f"[{b}] n={len(items)} cats={dict(list(cats.items())[:8])} img_sizes={sizes}")
        print("   prompt:", repr(items[0]["prompt"][:240]), "| gold_idx:", items[0]["gold_idx"], "| golds:", items[0]["golds"])
