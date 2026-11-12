"""6 个公开榜的规则判分（纯 Python，不依赖 torch）。`python bench_score.py` 跑自检。"""

import math
import re

SCORER_VERSION = "clean_v6_official_relaxed_v2"

_NUM = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*(%)?")
_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_YN_MAP = {"true": "yes", "false": "no", "y": "yes", "n": "no"}
_ANS_PREFIX = re.compile(r"^(?:the\s+)?(?:final\s+)?(?:answer|choice|option|conclusion)\s*(?:is)?\s*[:=-]?\s*", re.I)
_NEG_WORDS = {"not", "no", "never", "neither", "nor", "without", "isnt", "arent", "wasnt", "werent", "cannot", "cant", "dont", "doesnt", "didnt"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def norm(s):
    return re.sub(r"\.0+$", "", re.sub(r"[\s,]", "", str(s).strip().lower()))


def cnorm(s):
    return norm(s).strip(".;:!?\"'()")


def first_number_with_pct(s):
    s = str(s)
    mb = _BOXED.findall(s)
    if mb:
        s = mb[-1]
    s = _ANS_PREFIX.sub("", s.strip())
    s = re.sub(r"^\s*\([a-zA-Z0-9]\)\s*", "", s.replace(",", ""))
    m = _NUM.search(s)
    if not m:
        return None, False
    num = float(m.group(1))
    has_pct = bool(m.group(3))
    if m.group(2) is not None:
        den = float(m.group(2))
        if den == 0:
            return None, False
        num = num / den
    return num, has_pct


def first_number(s):
    val, _ = first_number_with_pct(s)
    return val


def _has_negation_before(text_low, target_low):
    idx = text_low.find(target_low)
    if idx < 0:
        return False
    prefix_words = re.findall(r"[a-z]+", text_low[:idx].replace("'", ""))
    return any(w in _NEG_WORDS for w in prefix_words[-3:])


def pick_choice(pred, choices):
    p = _ANS_PREFIX.sub("", pred.strip()).strip()
    if not p or not choices:
        return -1
    if choices == ["Yes", "No"]:
        low = cnorm(p)
        if low in ("yes", "true", "y", "correct", "right"):
            return 0
        if low in ("no", "false", "n", "incorrect", "wrong"):
            return 1
        words = re.findall(r"[a-z]+", p.lower())
        if words:
            if words[0] in ("yes", "true"):
                return 0
            if words[0] in ("no", "false"):
                return 1
        return -1
    m = re.fullmatch(r"\(?([A-Ha-h])\)?[.):]?", p) or re.match(r"\(?([A-Ha-h])[.):]\s*\S", p)
    if m:
        i = ord(m.group(1).upper()) - 65
        if 0 <= i < len(choices):
            return i
    pc = cnorm(p)
    exact = [i for i, c in enumerate(choices) if cnorm(c) == pc]
    if len(exact) == 1:
        return exact[0]
    p_low = p.lower()
    part = []
    for i, c in enumerate(choices):
        c_str = str(c).strip().lower()
        if len(cnorm(c)) >= 3 and re.search(r"(?<![a-z0-9])" + re.escape(c_str) + r"(?![a-z0-9])", p_low):
            if not _has_negation_before(p_low, c_str):
                part.append(i)
    return part[0] if len(part) == 1 else -1


def hit_mc(pred, choices, gold_idx):
    return gold_idx >= 0 and pick_choice(pred, choices) == gold_idx


def extract_answer_text(pred: str) -> str:
    """统一答案抽取：优先取 \\boxed{...}，否则取首个非空行并剥离标准前缀与末尾句号。"""
    s = str(pred or "").strip()
    mb = _BOXED.findall(s)
    if mb:
        s = mb[-1].strip()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    s = lines[0] if lines else ""
    s = _ANS_PREFIX.sub("", s).strip()
    return s.rstrip(".").strip()


def _parse_num_and_pct(t):
    """解析纯数值字符串，返回 (不除以100的原始数值, 是否显式带百分号 %)。"""
    t = str(t).strip().replace(",", "").replace("$", "")
    has_pct = t.endswith("%")
    if has_pct:
        t = t[:-1].strip()
    try:
        return float(t), has_pct
    except ValueError:
        return None, False


def _rel_ok(pred_val: float, gold_val: float, tol: float = 0.05) -> bool:
    if gold_val == 0.0:
        return abs(pred_val) <= 1e-6
    return abs(pred_val - gold_val) / abs(gold_val) <= tol


def hit_relaxed(pred, golds):
    """ChartQA 官方 relaxed correctness：数值相对误差 ≤5%（仅当预测与金标恰好一方显式带 % 时做单次 ×100/÷100 换算，绝不二次换算），非数值要求同义归一后精确匹配。"""
    p = extract_answer_text(pred)
    if not p:
        return False
    pf_exact, p_pct_exact = _parse_num_and_pct(p)
    pf_extracted, p_pct_extracted = first_number_with_pct(p)
    for g in golds:
        g_str = str(g).strip()
        gf, g_has_pct = _parse_num_and_pct(g_str)
        if gf is not None:
            if pf_exact is not None:
                pf, p_has_pct = pf_exact, p_pct_exact
            else:
                pf, p_has_pct = pf_extracted, p_pct_extracted
            if pf is None:
                continue
            if _rel_ok(pf, gf):
                return True
            if p_has_pct and not g_has_pct and _rel_ok(pf / 100.0, gf):
                return True
            if g_has_pct and not p_has_pct and _rel_ok(pf * 100.0, gf):
                return True
        else:
            gs = _YN_MAP.get(g_str.lower().rstrip("."), g_str.lower().rstrip("."))
            ps = _YN_MAP.get(p.lower().rstrip("."), p.lower().rstrip("."))
            if ps == gs:
                return True
    return False


def hit_ocr(pred, golds, qtype):
    hme = qtype == "Handwritten Mathematical Expression Recognition"

    def f(s):
        s = str(s).strip().replace("\n", " ")
        return s.replace(" ", "") if hme else s.lower()

    p = f(pred)
    return any(f(g) in p for g in golds)


def hit_geo(pred, golds):
    g = str(golds[0]).strip()
    try:
        gv = float(g.replace(",", ""))
    except ValueError:
        return norm(pred) == norm(g)
    pv = first_number(pred)
    return pv is not None and abs(pv - gv) <= max(0.05, 0.02 * abs(gv))


def hit_mv_free(pred, gold, answer_type, precision):
    if answer_type in ("integer", "float"):
        try:
            gv = float(str(gold).replace(",", ""))
        except ValueError:
            return norm(pred) == norm(gold)
        pv = first_number(pred)
        if pv is None:
            return False
        if answer_type == "integer":
            return int(round(pv)) == int(round(gv))
        d = int(precision) if precision is not None else 2
        return round(pv, d) == round(gv, d)
    return norm(pred) == norm(gold)


def score_item(bench, it, pred):
    pred = pred.strip()
    if bench in ("mmstar", "hallusion"):
        return hit_mc(pred, it["choices"], it["gold_idx"])
    if bench == "chartqa":
        return hit_relaxed(pred, it["golds"])
    if bench == "ocrbench":
        return hit_ocr(pred, it["golds"], it["cat"])
    if bench == "mathvista":
        if it["choices"]:
            return hit_mc(pred, it["choices"], it["gold_idx"])
        return hit_mv_free(pred, it["golds"][0], it.get("answer_type"), it.get("precision"))
    if bench == "geometry3k":
        return hit_geo(pred, it["golds"])
    raise ValueError(bench)


def aggregate(name, recs):
    n = len(recs)
    k = sum(r["hit"] for r in recs)
    lo, hi = wilson(k, n)
    cats = {}
    for r in recs:
        cats.setdefault(r["cat"], []).append(r["hit"])
    out = {"n": n, "hits": k, "acc": k / max(n, 1), "ci95": [lo, hi],
           "empty_rate": sum(1 for r in recs if not r["pred"].strip()) / max(n, 1),
           "by_cat": {c: {"n": len(v), "acc": sum(v) / len(v)} for c, v in sorted(cats.items())}}
    if name == "ocrbench":
        out["score_1000"] = 1000.0 * k / max(n, 1)
    return out


def selftest():
    ch = ["30°", "140°", "50°", "60°"]
    assert pick_choice("B", ch) == 1 and pick_choice("(C) 50°", ch) == 2 and pick_choice("d. 60°", ch) == 3
    assert pick_choice("140°", ch) == 1 and pick_choice("A cat sat", ["A cat sat", "x"]) == 0
    assert pick_choice("yes.", ["Yes", "No"]) == 0 and pick_choice("No", ["Yes", "No"]) == 1
    assert pick_choice("Answer: No", ["Yes", "No"]) == 1 and pick_choice("Answer: Yes", ["Yes", "No"]) == 0
    assert pick_choice("yesterday", ["Yes", "No"]) == -1
    assert hit_relaxed("14", ["14"]) and hit_relaxed("The answer is 14.5", ["14"]) and not hit_relaxed("15", ["14"])
    assert not hit_relaxed("1400", ["14"]) and not hit_relaxed("not blue", ["blue"])
    assert not hit_relaxed("100", ["1"]) and not hit_relaxed("10", ["0.1"])
    assert not hit_relaxed("24%", ["0.0024"]) and not hit_relaxed("100%", ["0.01"])
    assert not hit_relaxed("0.24", ["0.0024%"]) and not hit_relaxed("The answer is 24%", ["0.0024"])
    assert not hit_relaxed("The. The. No. The", ["No"])
    assert hit_relaxed("24%", ["0.24"]) and hit_relaxed("24%", ["24"]) and hit_relaxed("24", ["24%"]) and hit_relaxed("0.24", ["24%"])
    assert hit_relaxed("False", ["No"]) and hit_relaxed("True", ["Yes"])
    assert first_number("1e3") == 1000.0 and first_number("1/2") == 0.5
    assert not hit_mv_free("10.4", "10", "float", 2)
    assert hit_ocr("Centre", ["CENTRE"], "Regular Text Recognition")
    assert hit_geo("48", ["48"]) and hit_geo(r"\boxed{(a) 48}", ["48"]) and not hit_geo("52", ["48"])
    assert hit_mv_free("1000", "1000", "integer", None) and hit_mv_free("1.23", "1.2", "float", 1)
    print("bench_score SELFTEST OK")


if __name__ == "__main__":
    selftest()
