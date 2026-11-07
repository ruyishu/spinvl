"""汇总 results/ 下所有 JSON → FINAL_REPORT.md（缺什么就标 pending，随时可重复运行）。"""

import glob
import json
import os
import re
import statistics
import time

D = "/data/ibu-bi-ai-share-disk/zk/project/rose-distill"
R = f"{D}/results"


def jl(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        try:
            out[os.path.basename(f)[:-5]] = json.load(open(f))
        except Exception:
            pass
    return out


def ms(vals, nd=3, pct=False):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    f = (lambda x: f"{100*x:.1f}") if pct else (lambda x: f"{x:.{nd}f}")
    if len(vals) == 1:
        return f(vals[0])
    return f"{f(statistics.mean(vals))}±{f(statistics.pstdev(vals))}"


def get(d, *path):
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def sec_p1(L):
    p1 = jl(f"{R}/p1/p1_*.json")
    L.append("## 1. 叙事 1：MLLM 蒸馏教动态旋转（Geometry3K，8B 老师，3 seeds）\n")
    if not p1:
        L.append("_pending_\n"); return
    L.append("| 臂 | n | 验证 CE(配对) ↓ | 打乱−配对 gap ↑ | 首 token 准确率(val) ↑ | 整答案 tf-acc(val) ↑ | EM(val,60) ↑ |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ("ce", "kd", "rel"):
        rs = [v for k, v in p1.items() if re.fullmatch(rf"p1_{arm}_s\d", k)]
        if not rs:
            continue
        L.append(f"| {arm} | {len(rs)} | {ms([get(r,'val','correct') for r in rs])} | {ms([get(r,'val','gap') for r in rs])} | "
                 f"{ms([get(r,'val','tf_first') for r in rs], pct=True)}% | {ms([get(r,'val','tf_full') for r in rs], pct=True)}% | "
                 f"{ms([get(r,'val','em') for r in rs], pct=True)}% |")
    L.append("\n**官方 Geometry3K test（601 题，从未参与训练/选模）**\n")
    L.append("| 臂 | n | CE(配对) ↓ | gap ↑ | 首 token ↑ | 整答案 tf-acc ↑ | EM(150) ↑ |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ("ce", "kd", "rel"):
        rs = [v for k, v in p1.items() if re.fullmatch(rf"p1_{arm}_s\d", k) and v.get("test")]
        if rs:
            L.append(f"| {arm} | {len(rs)} | {ms([get(r,'test','correct') for r in rs])} | {ms([get(r,'test','gap') for r in rs])} | "
                     f"{ms([get(r,'test','tf_first') for r in rs], pct=True)}% | {ms([get(r,'test','tf_full') for r in rs], pct=True)}% | "
                     f"{ms([get(r,'test','em') for r in rs], pct=True)}% |")
    ab = {k: v for k, v in p1.items() if k.startswith("p1_abl_")}
    if ab:
        L.append("\n**超参消融（rel 臂，seed 0；对照 = p1_rel_s0）**\n")
        L.append("| 配置 | 验证 CE ↓ | gap ↑ | 首 token ↑ | test 首 token ↑ | test tf-full ↑ |")
        L.append("|---|---|---|---|---|---|")
        base = p1.get("p1_rel_s0")
        for name, v in ([("默认 (T=2, 7 层, b=64)", base)] if base else []) + [(k.replace("p1_abl_", ""), v) for k, v in sorted(ab.items())]:
            L.append(f"| {name} | {get(v,'val','correct'):.3f} | {get(v,'val','gap'):.3f} | {100*get(v,'val','tf_first'):.1f}% | "
                     f"{100*(get(v,'test','tf_first') or 0):.1f}% | {100*(get(v,'test','tf_full') or 0):.1f}% |")
    L.append("")


def sec_p2(L):
    p2 = jl(f"{R}/p2/p2_*.json")
    L.append("## 2. 数据规模曲线：蒸馏是否让小数据更省（mix 约 2.4 万题，人工标注短答案 VQA）\n")
    if not p2:
        L.append("_pending_\n"); return
    Ns = sorted({int(re.search(r"_n(\d+)_", k).group(1)) for k in p2})
    L.append("| N(训练样本) | 臂 | seeds | 验证 CE ↓ | gap ↑ | 首 token ↑ | 整答案 tf-acc ↑ |")
    L.append("|---|---|---|---|---|---|---|")
    table = {}
    for N in Ns:
        for arm in ("ce", "kd", "rel"):
            rs = [v for k, v in p2.items() if re.fullmatch(rf"p2_{arm}_n{N}_s\d", k)]
            if not rs:
                continue
            table[(N, arm)] = rs
            L.append(f"| {N} | {arm} | {len(rs)} | {ms([get(r,'val','correct') for r in rs])} | {ms([get(r,'val','gap') for r in rs])} | "
                     f"{ms([get(r,'val','tf_first') for r in rs], pct=True)}% | {ms([get(r,'val','tf_full') for r in rs], pct=True)}% |")
    # 样本效率：各臂达到「ce 在最大 N 的整答案准确率」所需的最小 N
    if table:
        Nmax = max(n for n, a in table if a == "ce") if any(a == "ce" for _, a in table) else None
        if Nmax:
            target = statistics.mean(get(r, "val", "tf_full") for r in table[(Nmax, "ce")])
            L.append(f"\n**样本效率**：以 ce@N={Nmax} 的整答案准确率 {100*target:.1f}% 为目标，各臂首次达到的最小 N：")
            for arm in ("ce", "kd", "rel"):
                hit = [n for n in Ns if (n, arm) in table and statistics.mean(get(r, 'val', 'tf_full') for r in table[(n, arm)]) >= target]
                L.append(f"- {arm}: {min(hit) if hit else '未达到'}")
    L.append("")


def sec_p3(L):
    p3 = jl(f"{R}/p3/p3_*.json")
    L.append("## 3. 叙事 2 · Stage 2A：前 K 层当内置视觉编码器（蒸馏 8B 视觉塔，指标=与老师特征的平均余弦）\n")
    if not p3:
        L.append("_pending_\n"); return
    L.append("| 前端 | 总可训练参数 | 其中静态适配参数 | 验证余弦 ↑ | Geometry3K-test 余弦(OOD) ↑ | 用时(min) |")
    L.append("|---|---|---|---|---|---|")
    for k, v in sorted(p3.items()):
        ex = v.get("extra_cos") or {}
        L.append(f"| {k.replace('p3_','')} | {v['trainable_M']:.1f}M | {v['static_M']:.2f}M | {v['val_cos']:.4f} | "
                 f"{(list(ex.values())[0] if ex else float('nan')):.4f} | {v['seconds']/60:.0f} |")
    L.append("\n> 对照解读：`none_K0`=只有 patch_embed+merger、不含任何 LLM 层；若 `rot` 明显高于它，说明被旋转的 LLM 前层确实在帮忙读图。\n")


def sec_p4(L):
    p4 = jl(f"{R}/p4/p4_*.json")
    L.append("## 4. 叙事 1+2 合体：拔掉外部视觉塔后的端到端表现（mix 验证子集）\n")
    if not p4:
        L.append("_pending_\n"); return
    L.append("| 配置 | 视觉来源 | 验证 CE ↓ | gap ↑ | 首 token ↑ | 整答案 tf-acc ↑ |")
    L.append("|---|---|---|---|---|---|")
    for k, v in sorted(p4.items()):
        for src, e in (v.get("eval") or {}).items():
            tag = "老师视觉塔(上限)" if src == "teacher" else "内置前端 v̂"
            L.append(f"| {k.replace('p4_','')} | {tag} | {e['correct']:.3f} | {e['gap']:.3f} | {100*e['tf_first']:.1f}% | {100*e['tf_full']:.1f}% |")
    L.append("")


def render_bench(L, b):
    for name, v in b.items():
        L.append(f"**效率（{name}，K={v['K']}，文本 Q={v['Q']}）**：TFLOPs / 延迟 ms\n")
        L.append("| 视觉 patch 数 | ours | rose(外挂塔+旋转) | keep(图片 token 全程保留) | llava(外挂塔+全程 token) |")
        L.append("|---|---|---|---|---|")
        for r in v["rows"]:
            def c(x):
                return f"{r[x]['tflops']:.2f} / {r[x]['ms']:.0f}" if "tflops" in r.get(x, {}) else "OOM/err"
            L.append(f"| {r['N_patches']} | {c('ours')} | {c('rose')} | {c('keep')} | {c('llava')} |")
        L.append(f"\n_{v.get('note','')}_\n")


def render_lang(L, lg):
    for name, v in lg.items():
        L.append(f"**语言保持（把前端并入权重的消融；GSM8K n={v['gsm_n']}，ARC-C n={v['arc_n']}）**\n")
        L.append("| 变体 | GSM8K | ARC-C |")
        L.append("|---|---|---|")
        for k, r in v["results"].items():
            L.append(f"| {k} | {100*r['gsm8k']:.1f}% | {100*r['arc_c']:.1f}% |")
        L.append("\n_`base` 即我们的文本通路（前端只作用于读图通路，文本推理与原模型逐位相同）。_\n")


def sec_p5(L):
    L.append("## 5. 效率与语言保持\n")
    b = jl(f"{R}/p5/bench*.json")
    if b:
        render_bench(L, b)
    else:
        L.append("_效率 pending_\n")
    lg = jl(f"{R}/p5/lang*.json")
    if lg:
        render_lang(L, lg)
    else:
        L.append("_语言保持 pending_\n")


def pc(x):
    return "—" if x is None else f"{100*x:.1f}%"


def arrow(a, b, pct=True):
    """before → after；任一缺失则整格为 —。"""
    if a is None or b is None:
        return "—"
    return f"{100*a:.1f}% → {100*b:.1f}%" if pct else f"{a:.1f} → {b:.1f}"


def sec_p6_opd(L):
    L.append("### 6.1 在线策略蒸馏（OPD）：治自回归跑偏\n")
    L.append("学生（CE 臂起点）自己 rollout，Qwen3-VL-8B 老师对**学生轨迹**逐 token 给分布做 KL。"
             "每格是「OPD 前 → OPD 后」；EM 是学生自己贪婪生成的整答案准确率，"
             "stop_rate 是自己在 max_new 内停下来的比例（EM 低而 stop_rate 也低 = 停不下来，不是答错）。\n")
    runs = {k: v for k, v in jl(f"{R}/p6/p6_opd_geo*_s?.json").items()
            if v.get("init_eval") and v.get("eval")}  # 旧代码跑出的结果没有 init_eval，直接排除
    if not runs:
        L.append("_pending_\n"); return
    cfgs = {}
    for name, v in sorted(runs.items()):
        a = v.get("args") or {}
        label = f"lr={a.get('lr', float('nan')):g}, ce_w={a.get('ce_weight', float('nan')):g}"
        cfgs.setdefault(label, []).append((name, v))
    L.append("| 配置 | 种子 | 评测集 | n | EM ↑ | 首 token(tf) ↑ | 整答案(tf) ↑ | stop_rate ↑ | 平均生成长度 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for cfg, items in cfgs.items():
        acc = {"val": [], "test": []}
        for name, v in items:
            for part, b, a in (("val", v["init_eval"], v["eval"]), ("test", v.get("init_test"), v.get("test"))):
                if not b or not a:
                    continue
                acc[part].append((b, a))
                L.append(f"| {cfg} | {name[-2:]} | {part} | {a.get('n_em')} | {arrow(b['em'], a['em'])} | "
                         f"{arrow(b['tf_first'], a['tf_first'])} | {arrow(b['tf_full'], a['tf_full'])} | "
                         f"{arrow(b['stop_rate'], a['stop_rate'])} | {arrow(b['mean_len'], a['mean_len'], pct=False)} |")
        for part, pairs in acc.items():
            if len(pairs) > 1:
                def avg(key, i, pairs=pairs):
                    return statistics.mean(p[i][key] for p in pairs)
                L.append(f"| {cfg} | 均值({len(pairs)}) | {part} | — | {arrow(avg('em',0), avg('em',1))} | "
                         f"{arrow(avg('tf_first',0), avg('tf_first',1))} | {arrow(avg('tf_full',0), avg('tf_full',1))} | "
                         f"{arrow(avg('stop_rate',0), avg('stop_rate',1))} | "
                         f"{arrow(avg('mean_len',0), avg('mean_len',1), pct=False)} |")
    mids = [(n, v["mid_eval"]) for n, v in sorted(runs.items()) if v.get("mid_eval")]
    if mids:
        L.append("\n训练中途（val 子集）EM 轨迹：" + "；".join(
            f"{n.replace('p6_opd_geo_', '')}: " + " → ".join(f"{m['step']}步 {pc(m['em'])}" for m in me)
            for n, me in mids))
    L.append("")


def _row(label, files, extra=""):
    """一行：同一配置的若干结果文件（多 seed）汇总成 mean±std。"""
    rs = list(files.values())
    if not rs:
        return None
    g = lambda *p: [get(r, *p) for r in rs]
    secs = [x for x in g("train_seconds") if x is not None]
    return (f"| {label} | {len(rs)} | {ms(g('val','correct'))} | {ms(g('val','gap'))} | "
            f"{ms(g('val','tf_first'), pct=True)}% | {ms(g('val','tf_full'), pct=True)}% | "
            f"{(statistics.mean(secs) / 60 if secs else float('nan')):.0f} | {extra} |")


HDR = ["| 配置 | seeds | 验证 CE ↓ | gap ↑ | 首 token ↑ | 整答案 tf ↑ | 训练用时(min) | 说明 |",
       "|---|---|---|---|---|---|---|---|"]


def sec_p6_proj(L):
    L.append("### 6.2 传统 Token 拼接（Projector）基线 vs 旋转注入\n")
    L.append("两者都冻结 Qwen3-1.7B、用同一份 mix、同一个固定切分的 val（与 P2 的 ce 臂是同一批样本）、同一步数公式。"
             "**`proj` 挂的是 8B 老师的外部视觉塔，视觉 token 全程进入 LLM 全部 28 层，"
             "属于「带外部视觉塔的传统拼接强基线」**；旋转注入同样用外部视觉塔的特征，"
             "但特征只用来生成旋转，LLM 里没有任何视觉 token。对比的是准确率与 FLOPs 开销，不是「拔掉视觉塔」。\n")
    rows = []
    for label, pat, extra in (
            ("proj N=6000", f"{R}/p6/p6_proj_n6000.json", "token 拼接，视觉 token 进全部层"),
            ("rotation(ce) N=6000", f"{R}/p2/p2_ce_n6000_s?.json", "旋转注入，lr 5e-4 / 2250 步"),
            ("proj N=21000", f"{R}/p6/p6_proj_n21000.json", "token 拼接，视觉 token 进全部层"),
            ("rotation(ce) N=21000", f"{R}/p2/p2_ce_n21000_s?.json", "旋转注入")):
        r = _row(label, jl(pat), extra)
        if r:
            rows.append(r)
    if not rows:
        L.append("_pending_\n"); return
    L += HDR + rows
    L.append("\n效率侧（FLOPs / 延迟）见 6.5 与第 5 节：旋转注入的 LLM 输入里没有视觉 token，proj 的序列长度随图片 patch 数线性增加。\n")


def sec_p6_scale(L):
    L.append("### 6.3 模型规模曲线（修正 1.7B 欠训练）\n")
    L.append("N=6000 同口径下，原先 1.7B 用 0.6B 的步数和学习率（2250 步、lr 5e-4）而旋转块数翻倍，疑似欠训练。"
             "这里补跑 lr 1e-3、4000 步的 1.7B，与 0.6B / 8B 并排。\n")
    rows = []
    for label, pat, extra in (
            ("Qwen3-0.6B (2250步, lr 5e-4)", f"{R}/p2s/p2s_q06_ce_n6000.json", "原口径"),
            ("Qwen3-1.7B (2250步, lr 5e-4)", f"{R}/p2/p2_ce_n6000_s?.json", "原口径"),
            ("Qwen3-1.7B (4000步, lr 1e-3)", f"{R}/p6/p6_lr_n6000_s?.json", "修正后"),
            ("Qwen3-8B (2250步, lr 5e-4)", f"{R}/p2s/p2s_q8b_ce_n6000.json", "原口径")):
        r = _row(label, jl(pat), extra)
        if r:
            rows.append(r)
    if not rows:
        L.append("_pending_\n"); return
    L += HDR + rows
    em = [(lab, [get(v, "val", "em") for v in jl(pat).values()])
          for lab, pat in (("0.6B", f"{R}/p2s/p2s_q06_ce_n6000.json"), ("1.7B 原", f"{R}/p2/p2_ce_n6000_s?.json"),
                           ("1.7B 修正", f"{R}/p6/p6_lr_n6000_s?.json"), ("8B", f"{R}/p2s/p2s_q8b_ce_n6000.json"))]
    em = [(lab, ms(v, pct=True)) for lab, v in em if any(x is not None for x in v)]
    if em:
        L.append("\n生成式 EM（n=100）：" + "；".join(f"{lab} {v}%" for lab, v in em) + "\n")


def sec_p6_rots(L):
    L.append("### 6.4 内置前端：旋转 + 逐通道对角缩放（rots）\n")
    L.append("`rots`：W' = W·R·diag(d)，在静态正交旋转上加逐通道缩放，补上正交矩阵改不了奇异值的短板。"
             "2A 的余弦与端到端反相关，只当热身；真指标是下面的端到端联合训练。\n")
    p3 = {**jl(f"{R}/p3/p3_*.json"), **jl(f"{R}/p6/p6_rots_K4.json")}
    names = [k for k in ("p3_none_K0", "p3_rot_K4", "p6_rots_K4", "p3_full_K4") if k in p3]
    if names:
        L.append("**Stage 2A（与 8B 视觉塔特征的余弦，仅供参考）**\n")
        L.append("| 前端 | 总可训练参数 | 静态适配参数 | 验证余弦 | Geometry3K-test 余弦 | 用时(min) |")
        L.append("|---|---|---|---|---|---|")
        for k in names:
            v, ex = p3[k], p3[k].get("extra_cos") or {}
            L.append(f"| {k.split('_',1)[1]} | {v['trainable_M']:.1f}M | {v['static_M']:.2f}M | {v['val_cos']:.4f} | "
                     f"{(list(ex.values())[0] if ex else float('nan')):.4f} | {v['seconds']/60:.0f} |")
        L.append("")
    p4 = {**jl(f"{R}/p4/p4_*.json"), **jl(f"{R}/p6/p6_rots_K4_joint.json")}
    rows = []
    for k, tag in (("p4_oracle", "老师视觉塔(上限)"), ("p4_none_K0_joint", "none K=0"), ("p4_rot_K4_joint", "rot K=4"),
                   ("p6_rots_K4_joint", "**rots K=4**"), ("p4_full_K4_joint", "full K=4")):
        v = p4.get(k)
        if not v:
            continue
        src = "teacher" if k == "p4_oracle" else "front"
        e = (v.get("eval") or {}).get(src)
        if e:
            start = "ce" if k.startswith("p6") else "rel"
            rows.append(f"| {tag} | {start} | {e['correct']:.3f} | {e['gap']:.3f} | "
                        f"{100*e['tf_first']:.1f}% | {100*e['tf_full']:.1f}% |")
    if rows:
        L.append("**端到端（拔掉外部视觉塔，front→v̂→VRMG；mix 验证子集）**\n")
        L.append("| 前端 | VRMG 起点 | 验证 CE ↓ | gap ↑ | 首 token ↑ | 整答案 tf ↑ |")
        L.append("|---|---|---|---|---|---|")
        L += rows
        L.append("\n_注意：rots 的 VRMG 起点是 ce 臂，p4 各行是 rel 臂；两种起点的 CE 绝对值不严格可比，"
                 "优先看首 token / 整答案准确率。_\n")
    if not names and not rows:
        L.append("_pending_\n")


def sec_p6_eff(L):
    L.append("### 6.5 效率与语言保持（补跑 K=2）\n")
    b, lg = jl(f"{R}/p6/bench*.json"), jl(f"{R}/p6/lang*.json")
    if b:
        render_bench(L, b)
    if lg:
        render_lang(L, lg)
    if not b and not lg:
        L.append("_pending_\n")


def sec_p6(L):
    L.append("## 6. 后续补充实验（2026-11-05 追加）\n")
    for f in (sec_p6_opd, sec_p6_proj, sec_p6_scale, sec_p6_rots, sec_p6_eff):
        try:
            f(L)
        except Exception as e:
            L.append(f"_{f.__name__} 汇总出错: {type(e).__name__}: {e}_\n")


SPECIAL_OUT = {"p5_lang": "p5/lang_K4.json", "p5_bench": "p5/bench_K4.json",
               "p6_lang": "p6/lang_K2.json", "p6_bench": "p6/bench_K2.json",
               "p6smoke_opd_v2": "p6/smoke_opd_v2.json"}


def has_result(job):
    """任务名对应的结果 JSON 是否已落盘（重启 worker 时 adopted=[] 会留下「failed 记录 + 实际跑完」的任务）。"""
    if job in SPECIAL_OUT:
        return os.path.exists(f"{R}/{SPECIAL_OUT[job]}")
    return bool(glob.glob(f"{R}/*/{job}.json"))


def sec_queue(L):
    q = f"{D}/queue"
    done = len(glob.glob(f"{q}/done/*.json"))
    failed = sorted(os.path.basename(x)[:-5] for x in glob.glob(f"{q}/failed/*.json"))
    stale = [n for n in failed if has_result(n)]
    real = [n for n in failed if n not in stale]
    L.append(f"## 队列状态\n\n本报告的表格都是直接扫 `results/` 下的 JSON 生成的，不依赖 `queue/done`。\n")
    L.append(f"队列记录完成 {done} 个任务；failed 记录 {len(failed)} 个。")
    if stale:
        L.append(f"- 其中 {len(stale)} 个 failed 记录**已有结果 JSON**（训练实际跑完，队列状态是陈旧的），结果已计入上面的表格：" + ", ".join(stale))
    L.append(f"- 真正失败（无结果 JSON）{len(real)} 个：" + (", ".join(real) if real else "无") + "\n")


def main():
    L = [f"# ROSE-Distill 实验总报告（自动生成 {time.strftime('%Y-%m-%d %H:%M')}）\n",
         "学生 Qwen3-1.7B（全程冻结，仅训练 VRMG / 前端）；老师 Qwen3-VL-8B-Instruct。\n"]
    for f in (sec_p1, sec_p2, sec_p3, sec_p4, sec_p5, sec_p6, sec_queue):
        try:
            f(L)
        except Exception as e:
            L.append(f"_{f.__name__} 汇总出错: {e}_\n")
    open(f"{D}/FINAL_REPORT.md", "w").write("\n".join(L))
    print("\n".join(L)[:6000])


if __name__ == "__main__":
    main()
