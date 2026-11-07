"""p10 冒烟闸门：检查产物的数值（不只是没崩）。通过退出码 0，失败退出码 1 并打印原因。
用法：
  p10_gate.py opd  <opd.json>                 OPD 结果：指标有限、训练后 tf_first 没有塌
  p10_gate.py json <x.json> [<a.b.c> ...]     产物可读，且给定路径上的数都是有限数
  p10_gate.py eval <bench.json> <expect_n>    评测结果：每榜题数=expect_n、分数有限、空输出率<0.9
"""

import json
import math
import sys


def fin(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def dig(j, path):
    for k in path.split("."):
        j = j[k]
    return j


def main():
    mode, js = sys.argv[1], sys.argv[2]
    j = json.load(open(js))
    errs = []
    if mode == "opd":
        a, b = j["init_eval"], j["eval"]
        for k in ("tf_first", "tf_full", "em"):
            if not (fin(a.get(k)) and fin(b.get(k))):
                errs.append(f"{k} 非有限数 init={a.get(k)} final={b.get(k)}")
        if not errs and b["tf_first"] < 0.5 * a["tf_first"]:
            errs.append(f"tf_first {a['tf_first']:.3f} -> {b['tf_first']:.3f} 塌了")
    elif mode == "json":
        for p in sys.argv[3:]:
            try:
                v = dig(j, p)
            except Exception as e:
                errs.append(f"{p} 取不到: {e!r}")
                continue
            if not fin(v):
                errs.append(f"{p}={v} 非有限数")
    elif mode == "eval":
        n0 = int(sys.argv[3])
        if not j.get("benches"):
            errs.append("benches 为空")
        for b, r in j.get("benches", {}).items():
            if r["n"] != n0:
                errs.append(f"{b}: n={r['n']} != {n0}")
            if not fin(r["acc"]):
                errs.append(f"{b}: acc={r['acc']} 非有限数")
            if r["empty_rate"] > 0.9:
                errs.append(f"{b}: 空输出率 {r['empty_rate']:.2f}")
    else:
        errs.append(f"未知模式 {mode}")
    print("GATE", mode, js, ("FAIL: " + "; ".join(errs)) if errs else "PASS", flush=True)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
