"""6 个公开榜统一评测（MMStar / HallusionBench / ChartQA / OCRBench / MathVista testmini / Geometry3K test）。"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_data as bd  # noqa: E402
import bench_score as bs  # noqa: E402

ROOT = bd.B + "/zk/project/rose-distill"


def _item_img_key(it, idx):
    raw = it.get("img_bytes")
    if raw is None and it.get("img_path") and os.path.exists(it["img_path"]):
        raw = open(it["img_path"], "rb").read()
    if raw is not None:
        sha = hashlib.sha256(raw).hexdigest()
        try:
            import io
            import numpy as np
            from PIL import Image
            im = Image.open(io.BytesIO(raw)).convert("L").resize((9, 8), Image.BILINEAR)
            g = np.asarray(im, dtype=np.int16)
            bits = (g[:, 1:] > g[:, :-1]).flatten()
            dh = 0
            for b in bits:
                dh = (dh << 1) | int(b)
            return f"dh:{dh:016x}"
        except Exception:
            return f"sha:{sha}"
    return f"id:{it.get('id', idx)}"


def build_derangement_by_image(items, shift_seed=1):
    """按图像内容 SHA256 分组构造严格异图置换（保證每道题换到的图与原图 SHA256 100% 不同）。"""
    n = len(items)
    if n <= 1:
        return list(range(n)), 0
    keys = [_item_img_key(it, idx) for idx, it in enumerate(items)]
    groups = defaultdict(list)
    for idx, k in enumerate(keys):
        groups[k].append(idx)
    if len(groups) <= 1:
        return list(range(n)), n
    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), keys[g[0]]))
    max_g = max(len(g) for g in sorted_groups)
    assert max_g * 2 <= n, f"最大同图组 {max_g} 超过总题数一半 {n}，无法构造零碰撞置换"
    ordered = [idx for g in sorted_groups for idx in g]
    shift = max_g + (shift_seed % max(1, n - 2 * max_g + 1))
    perm = [0] * n
    for pos, orig_idx in enumerate(ordered):
        perm[orig_idx] = ordered[(pos + shift) % n]
    same_cnt = sum(1 for i in range(n) if keys[perm[i]] == keys[i])
    assert same_cnt == 0, f"异图置换仍存在 {same_cnt} 道同图题！"
    return perm, same_cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tower", "tower_free", "proj", "proj_pool32", "proj_fastv32", "teacher"])
    ap.add_argument("--ckpt")
    ap.add_argument("--name")
    ap.add_argument("--model", default=f"{bd.B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--benches", default=",".join(bd.BENCHES))
    ap.add_argument("--limit", type=int, default=0, help="每榜只测前 N 题（冒烟用），0=全量")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-pixels", type=int, default=392 * 392)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--image-ctrl", choices=["none", "shuffle", "blank"], default="none")
    ap.add_argument("--force-gen", action="store_true", help="即使是选择题也走贪心自由生成，并统计 EOS 结束率/复读率/生成长度")
    ap.add_argument("--flags", default="", help="手动开结构开关，如 --multifreq,--hybrid（默认从 ckpt 的 args 读）")
    ap.add_argument("--prompt-alpha", type=float, default=None, help="Prompt 段旋转强度掩码（None=按 ckpt 配置）")
    ap.add_argument("--hyb-bound", type=float, default=None, help="hybrid 残差相对范数上限（None=按 ckpt 配置）")
    ap.add_argument("--out-dir", default=ROOT + "/results/bench_eval")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check-data", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return bs.selftest()
    if args.check_data:
        return bd.check_data(args.limit)
    assert args.mode and args.ckpt and args.name, "需要 --mode --ckpt --name"

    import torch
    from bench_models import Runner

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    run = Runner(args.mode, args.ckpt, args.model, flags=tuple(x for x in args.flags.split(",") if x),
                 prompt_alpha=args.prompt_alpha, hyb_bound=args.hyb_bound)
    print(f"loaded mode={args.mode} arch={run.arch} in {time.time() - t0:.0f}s", flush=True)
    res, pf = {}, open(f"{args.out_dir}/{args.name}.preds.jsonl", "w")
    for b in args.benches.split(","):
        items = bd.load_bench(b, args.limit)
        perm, same_img_cnt = build_derangement_by_image(items) if args.image_ctrl == "shuffle" else (list(range(len(items))), 0)
        recs, tb = [], time.time()
        for i in range(0, len(items), args.bs):
            chunk = items[i : i + args.bs]
            src = [items[perm[i + k]] for k in range(len(chunk))] if args.image_ctrl == "shuffle" else chunk
            if args.image_ctrl == "blank":
                from PIL import Image
                imgs = []
                for it in src:
                    im_real = bd.get_image(it, args.max_pixels)
                    sz = im_real.size if im_real is not None else (224, 224)
                    imgs.append(Image.new("RGB", sz, (0, 0, 0)))
            else:
                imgs = [bd.get_image(it, args.max_pixels) for it in src]
            prompts = [it["prompt"] for it in chunk]
            try:
                preds = run.generate(prompts, imgs, args.max_new, items=chunk, bench=b, force_gen=args.force_gen)
                stats_list = list(getattr(run, "last_gen_stats", None) or [{}] * len(preds))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                preds, stats_list = [], []
                for p, im, it in zip(prompts, imgs, chunk):
                    p_one = run.generate([p], [im], args.max_new, items=[it], bench=b, force_gen=args.force_gen)[0]
                    st_one = (getattr(run, "last_gen_stats", None) or [{}])[0]
                    preds.append(p_one)
                    stats_list.append(st_one)
            assert len(preds) == len(chunk) and len(stats_list) == len(chunk), "预测或统计条数与输入 chunk 不一致"
            for it, p, st in zip(chunk, preds, stats_list):
                hit = int(bs.score_item(b, it, p))
                recs.append({"id": it["id"], "cat": it["cat"], "pred": p, "hit": hit, "gold_idx": it.get("gold_idx", -1), **st})
                pf.write(json.dumps({"bench": b, "id": it["id"], "pred": p, "gold": it["golds"], "hit": hit, **st},
                                    ensure_ascii=False) + "\n")
            if (i // args.bs) % 25 == 0:
                print(f"[{b}] {len(recs)}/{len(items)} acc={sum(r['hit'] for r in recs) / len(recs):.3f} "
                      f"{time.time() - tb:.0f}s", flush=True)
        res[b] = bs.aggregate(b, recs)
        if args.image_ctrl == "shuffle":
            res[b]["shuffle_same_image_count"] = same_img_cnt
        if b == "hallusion" or any(r.get("pred") in ("Yes", "No") for r in recs):
            yn_recs = [r for r in recs if r.get("pred") in ("Yes", "No")]
            if yn_recs:
                res[b]["pred_yes_rate"] = sum(1 for r in yn_recs if r["pred"] == "Yes") / len(yn_recs)
                res[b]["pred_no_rate"] = sum(1 for r in yn_recs if r["pred"] == "No") / len(yn_recs)
            yes_g = [r["hit"] for r in recs if r.get("gold_idx") == 0]
            no_g = [r["hit"] for r in recs if r.get("gold_idx") == 1]
            if yes_g and no_g:
                res[b]["yes_acc"] = sum(yes_g) / len(yes_g)
                res[b]["no_acc"] = sum(no_g) / len(no_g)
                res[b]["balanced_acc"] = 0.5 * (res[b]["yes_acc"] + res[b]["no_acc"])
        mc_recs = [r for r in recs if "cand_logits" in r and r.get("gold_idx", -1) >= 0]
        if mc_recs:
            by_nc = defaultdict(list)
            for r in mc_recs:
                by_nc[len(r["cand_logits"])].append(r)
            cal_hits = 0
            for nc, grp in by_nc.items():
                mean_lg = [sum(r["cand_logits"][c] for r in grp) / len(grp) for c in range(nc)]
                for r in grp:
                    adj = [r["cand_logits"][c] - mean_lg[c] for c in range(nc)]
                    pred_idx = int(max(range(nc), key=lambda c: adj[c]))
                    cal_hits += int(pred_idx == r["gold_idx"])
            res[b]["cal_acc"] = cal_hits / len(mc_recs)
            res[b]["cal_acc_n"] = len(mc_recs)
        if any("eos_stop" in r for r in recs):
            gr = [r for r in recs if "eos_stop" in r]
            res[b]["eos_stop_rate"] = sum(r["eos_stop"] for r in gr) / len(gr)
            res[b]["avg_gen_tokens"] = sum(r["gen_tokens"] for r in gr) / len(gr)
            res[b]["rep3_rate"] = sum(r["rep3"] for r in gr) / len(gr)
        res[b]["seconds"] = time.time() - tb
        pf.flush()
        print(f"== {b}: n={res[b]['n']} acc={res[b]['acc']:.4f} ci95={[round(x, 4) for x in res[b]['ci95']]} "
              f"empty={res[b]['empty_rate']:.2f}", flush=True)
    out = {"name": args.name, "mode": args.mode, "ckpt": args.ckpt, "model": args.model, "arch": run.arch,
           "args": vars(args), "benches": res, "seconds": time.time() - t0,
           "peak_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
           "note": "对齐训练格式的规则判分（选择题/判断题用候选对数似然，开放题贪心解码+防复读）；图片缩放到 ≤max_pixels"}
    tmp = f"{args.out_dir}/{args.name}.json.tmp"
    json.dump(out, open(tmp, "w"), indent=1, ensure_ascii=False)
    os.replace(tmp, f"{args.out_dir}/{args.name}.json")
    print("EVAL_DONE", args.name, flush=True)


if __name__ == "__main__":
    main()
