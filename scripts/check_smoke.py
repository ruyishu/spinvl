"""冒烟结果的闸门：plumbing 没问题才写 <结果>.ok，正式任务 needs 这个 .ok 才会起跑。

文件存在只能说明没崩。上一轮冒烟「通过」了，正式跑却把 tf_first 训到 0（学生答案 token 和 pad 列错位、KL 一次没生效），
所以这里检查数值。只拦 plumbing：结果字段缺失、非有限 loss、KL 一次没生效、老师前缀与学生 prompt 不一致、起点指标异常。
训练后 tf_first 下滑属于实验结果（优化设置/老师与金标不一致），只打 SMOKE_WARN，不拦；正式任务会记录 init_eval→eval 与 mid_eval。
用法：python check_smoke.py opd|proj <result.json>
"""

import json
import math
import sys

WARN = []


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def check_opd(r):
    ie, ev = r.get("init_eval"), r.get("eval")
    if not ie or not ev:
        return ["结果里缺 init_eval / eval（不是新版 train_opd 跑出来的）"]
    errs = [f"{k} 不是有限数" for k in ("tf_first", "tf_full", "em", "stop_rate", "mean_len")
            if not (finite(ie.get(k)) and finite(ev.get(k)))]
    if errs:
        return errs
    if ie["tf_first"] < 0.5:
        errs.append(f"起点 tf_first={ie['tf_first']:.3f} 过低，VRMG 检查点可能没载对")
    if ev["tf_first"] < ie["tf_first"] - 0.25:
        WARN.append(f"训练后 tf_first {ie['tf_first']:.3f} → {ev['tf_first']:.3f} 明显下滑（不拦，看正式任务的 mid_eval）")
    tail = r.get("loss_tail") or []
    if not tail:
        errs.append("没有训练记录")
    else:
        if any(not finite(t[0]) for t in tail):
            errs.append("出现非有限 loss")
        if sum(t[1] for t in tail) <= 0:
            errs.append("KL 项一次都没生效")
        if sum(t[3] for t in tail) > 0:
            errs.append("老师前缀与学生 prompt 的 token 不一致")
    return errs


def check_proj(r):
    v = r.get("val") or {}
    errs = [f"{k} 不是有限数" for k in ("correct", "shuffled", "no_image", "tf_first", "tf_full")
            if not finite(v.get(k))]
    if not r.get("n_val") or not r.get("n_train"):
        errs.append("缺 n_val / n_train（不是新版 train_proj 跑出来的）")
    return errs


if __name__ == "__main__":
    kind, path = sys.argv[1], sys.argv[2]
    errs = {"opd": check_opd, "proj": check_proj}[kind](json.load(open(path)))
    for w in WARN:
        print("SMOKE_WARN", w)
    if errs:
        print("SMOKE_CHECK_FAIL", *errs, sep="\n  ")
        sys.exit(1)
    ok = path[: -len(".json")] + ".ok"
    open(ok, "w").write("ok\n")
    print("SMOKE_CHECK_OK ->", ok)
