"""在线策略蒸馏（OPD）：学生在自己的 rollout 轨迹上接受老师逐 token 纠正。

与离线 KD 的区别：离线 KD 只在「标准答案前缀」上对齐，治不了自回归跑偏
（Geometry3K 上首 token 97% 但整答案 EM 只有 3%）。OPD 每步让学生先自己生成一段回答，
再让 Qwen3-VL-8B 老师对这段**学生轨迹**做一次前向，取老师分布与学生的 KL。

数据流：
    prompt ──学生(冻结 LLM + VRMG 动态旋转)──▶ 自回归 rollout ŷ（保留真实 token id，不 decode 再分词）
    image + prompt + ŷ ──老师(Qwen3-VL-8B)──▶ 每个位置的 p_T
    loss = KL(p_T ‖ p_θ) · T² + ce_weight · CE(金标 prompt+答案+EOS，同 P1 的 collate)

评测：teacher-forced 首 token / 整答案准确率，加上学生自己贪婪生成的整答案 EM
（归一化与 train_final.eval_accuracy 完全相同）。训练前（step 0）先评一次记入 result["init_eval"]，
训练后的结果在 result["eval"]，二者之差就是 OPD 带来的提升。
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import imgdata  # noqa: E402
import train_final as tf  # noqa: E402
from provenance import capture_run_inputs, build_run_manifest, snapshot_backbone_hashes, verify_frozen_backbone_and_optimizer  # noqa: E402
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
VISION_PREFIX = "<|vision_start|><|image_pad|><|vision_end|>"


def pad_vis(v_list, dev):
    n = max(v.shape[0] for v in v_list)
    out = torch.zeros(len(v_list), n, v_list[0].shape[-1], device=dev, dtype=torch.float32)
    m = torch.zeros(len(v_list), n, device=dev)
    for i, v in enumerate(v_list):
        out[i, : v.shape[0]] = v
        m[i, : v.shape[0]] = 1
    return out, m


def stop_ids(tok):
    """生成时的停止 token：eos（<|im_end|>）与 pad（<|endoftext|>）。"""
    s = {tok.eos_token_id, tok.pad_token_id}
    return sorted(x for x in s if x is not None)


def cut_at_stop(row, stops, include_stop=False):
    """截取真实生成 token；在线 KL 用 include_stop=True 保留自然 EOS，评测解码默认去掉。"""
    out = []
    for t in row.tolist():
        if t in stops:
            if include_stop:
                out.append(t)
            break
        out.append(t)
    return out


@torch.no_grad()
def tf_accuracy(model, registry, tok, data, idx, dev, n_acc=100):
    """给定正确前缀的首 token / 整答案准确率（与其它阶段同口径）。"""
    model.eval()
    first = full = n = 0
    for i in idx[:n_acc]:
        tid = tok(data["prompts"][i] + " " + data["answers"][i])["input_ids"]
        a0, na = data["a_rel_start"][i], data["n_answer"][i]
        vis = data["vfeat"][i].unsqueeze(0).to(dev).float()
        vm = torch.ones(1, vis.shape[1], device=dev)
        with rotation_context_with(registry, vis, vm, ans_start=[a0], text_mask=torch.ones(1, len(tid), device=dev)):
            lg = model(input_ids=torch.tensor([tid], device=dev)).logits[0].float()
        preds = [lg[a0 - 1 + k].argmax().item() for k in range(na)]
        first += int(preds[0] == tid[a0])
        full += int(preds == tid[a0 : a0 + na])
        n += 1
    model.train()
    return {"tf_first": first / max(n, 1), "tf_full": full / max(n, 1)}


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_answer(text):
    """CoT 模式：优先取 \boxed{...}，否则取最后一行文本。"""
    m = BOXED_RE.findall(text)
    if m:
        return m[-1]
    return text.strip().splitlines()[-1] if text.strip() else ""


@torch.no_grad()
def greedy_em(model, registry, tok, data, idx, dev, n=1000, max_new=24, boxed=False):
    """统一验证集贪婪生成评测协议（与 train_final.eval_greedy_stats / Tower-Free 完全一致）。"""
    if not boxed:
        return tf.eval_greedy_stats(model, registry, tok, idx, data, dev, n=n, max_new=max_new, bs=8)
    model.eval()
    stops = set(stop_ids(tok))
    hit = stopped = tot_len = rep3_cnt = cnt = 0
    for i in idx[:n]:
        ids = tok(data["prompts"][i])["input_ids"]
        vis = data["vfeat"][i].unsqueeze(0).to(dev).float()
        vm = torch.ones(1, vis.shape[1], device=dev)
        with rotation_context_with(registry, vis, vm, text_mask=torch.ones(1, len(ids), device=dev)):
            out = model.generate(input_ids=torch.tensor([ids], device=dev), max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 eos_token_id=sorted(stops))
        gen = out[0, len(ids):]
        real = cut_at_stop(gen, stops)
        pred = tok.decode(real, skip_special_tokens=True)
        hit += int(tf._norm(extract_answer(pred)) == tf._norm(extract_answer(data["answers"][i])))
        stopped += int(len(real) < len(gen))
        tot_len += len(real)
        ng3 = [tuple(real[m : m + 3]) for m in range(max(0, len(real) - 2))]
        rep3_cnt += int(len(ng3) >= 2 and len(set(ng3)) < len(ng3))
        cnt += 1
    model.train()
    return {"em": hit / max(cnt, 1), "stop_rate": stopped / max(cnt, 1),
            "mean_len": tot_len / max(cnt, 1), "rep3_rate": rep3_cnt / max(cnt, 1), "em_n": cnt}


def evaluate(model, registry, tok, data, idx, dev, n_tf, n_em, max_new, boxed=False):
    """teacher-forced 准确率 + 学生自己贪婪生成的 EM，合成一个字典。"""
    r = tf_accuracy(model, registry, tok, data, idx, dev, n_tf)
    r.update(greedy_em(model, registry, tok, data, idx, dev, n_em, max_new, boxed))
    r["n_tf"], r["n_em"] = min(n_tf, len(idx)), min(n_em, len(idx))
    return r


def fmt(d):
    return " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def build_teacher_batch(proc, prompts, pids, gids, imgs, dev, pad_id):
    """老师侧输入：processor 编码 VISION_PREFIX+prompt（左填充），后面直接接学生 rollout 的真实 token id。

    学生与老师共用 Qwen3 词表，rollout 不需要 decode 再分词，逐 token 精确对齐：
    第 j 个 rollout token 位于第 Lp+j 列，预测它的 logits 在第 Lp-1+j 列（Lp=前缀含左填充的列数）。
    bad[k]=True 表示老师前缀末尾与学生 prompt 的 token 不一致（理论上不会发生，发生时该样本不进 KL）。
    """
    concise_inst = "Answer concisely with only the final word, number, or option letter.\n"
    texts = [VISION_PREFIX + concise_inst + p for p in prompts]
    if getattr(build_teacher_batch, "boxed", False):
        inst = ("Solve the geometry problem with a short derivation, "
                "then output the final answer as \\boxed{...}.\n")
        texts = [VISION_PREFIX + inst + p for p in prompts]
    enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True)
    enc = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in enc.items()}
    ids, mask = enc["input_ids"], enc["attention_mask"]
    assert bool((mask[:, -1] == 1).all()), "老师侧不是左填充，rollout 的拼接位置会错位"
    Lp, maxa = ids.shape[1], max([len(g) for g in gids] + [1])
    bad = [ids[k, Lp - len(p):].tolist() != list(p) for k, p in enumerate(pids)]
    tail = torch.full((len(gids), maxa), pad_id, dtype=ids.dtype, device=dev)
    tmask = torch.zeros(len(gids), maxa, dtype=mask.dtype, device=dev)
    for k, g in enumerate(gids):
        if g:
            tail[k, : len(g)] = torch.tensor(g, device=dev)
            tmask[k, : len(g)] = 1
    enc["input_ids"] = torch.cat([ids, tail], dim=1)
    enc["attention_mask"] = torch.cat([mask, tmask], dim=1)
    for k, v in list(enc.items()):
        if k not in ("input_ids", "attention_mask") and torch.is_tensor(v) and v.dim() == 2 and v.shape == ids.shape:
            enc[k] = torch.cat([v, torch.zeros(v.shape[0], maxa, dtype=v.dtype, device=dev)], dim=1)
    return enc, Lp, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--source", action="append", required=True, help="meta|manifest|img_root")
    ap.add_argument("--test-data", default="", help="官方 test 的 meta（如 geo_8b_test.pt）；给了就在训练前后各评一次")
    ap.add_argument("--vrmg-ckpt", required=True, help="学生起点：CE 臂的 *_pt.pt")
    ap.add_argument("--model", default=f"{B}/zk/model/Qwen3-1.7B")
    ap.add_argument("--teacher", default=imgdata.TEACHER)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积：每个优化步用 accum 个 bs 大小的微批")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--boxed", action="store_true", help="CoT 模式：抽 \boxed{} 评准确率，老师 prompt 要求逐步推导")
    ap.add_argument("--prompt-alpha", type=float, default=1.0, help="Prompt 段旋转强度掩码（<1 保留原 LLM 读题能力）")
    ap.add_argument("--hyb-bound", type=float, default=0.0, help="hybrid 残差相对范数上限（>0 防长序列激活爆炸）")
    ap.add_argument("--multifreq", action="store_true")
    ap.add_argument("--tokrot", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--qcond", action="store_true", help="问题条件化旋转：把题目文本池化向量叠到 VRMG queries")
    ap.add_argument("--direct-hyb", action="store_true", help="hybrid 直接在 256 维对全图视觉 cell 做无损交叉注意力")
    ap.add_argument("--no-rot", action="store_true", help="旋转恒等映射 R=I，仅保留视觉 cross-attention")
    ap.add_argument("--rot-bound", type=float, default=0.0, help="李代数切空间有界正交旋转半径（>0 启用）")
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--min-new", type=int, default=1, help="只防空输出，不硬拽废话")
    ap.add_argument("--kd-temp", type=float, default=1.0)
    ap.add_argument("--ce-weight", type=float, default=0.2, help="金标 prompt+答案+EOS 的 CE 锚点权重，0 关闭")
    ap.add_argument("--seq", type=int, default=160, help="金标 CE 的序列长度（同 P1）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--tune-backbone", action="store_true", help="已禁用：当前主线强制要求原始 LLM 全程冻结")
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--n-eval", type=int, default=1000, help="val 上评测的样本数（分层抽取 1000 题）")
    ap.add_argument("--n-test", type=int, default=0, help="test 上评测的样本数，0=全部")
    ap.add_argument("--eval-every", type=int, default=0, help="训练中途评测间隔，0=steps//3（至少 100）")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.tune_backbone:
        raise ValueError("当前冻结主线严禁启用 --tune-backbone！原始 LLM 必须全程 100% 冻结。")
    capture_run_inputs(vars(args))
    torch.manual_seed(args.seed)
    dev = "cuda"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    logf = open(args.out + ".log", "a")

    def log(m):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    paths = {}
    for s in args.source:
        paths.update(imgdata.parse_source(s)[1])
    data = tf.load_data(args.data)
    n = len(data["ids"])
    train, val, split_info = tf.split_train_val_by_image(data, paths, seed=args.seed)
    train = [i for i in train if data["ids"][i] in paths]
    val = [i for i in val if data["ids"][i] in paths]
    assert train and val, "按图切分后可用 train/val 为空"
    val_em_1000, val_1000_sha1 = tf.select_stratified_val_1000(val, data, n_target=args.n_eval, seed=0)
    tdata, tidx = None, []
    if args.test_data:
        tdata = tf.load_data(args.test_data)
        tidx = list(range(len(tdata["ids"])))
        if args.n_test:
            tidx = tidx[: args.n_test]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    stops = stop_ids(tok)
    build_teacher_batch.boxed = args.boxed
    proc = AutoProcessor.from_pretrained(args.teacher, trust_remote_code=True)
    proc.tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True)
    ck = torch.load(args.vrmg_ckpt, map_location=dev)
    a1 = ck["args"]
    layers = [int(x) for x in a1["layers"].split(",") if x]
    pa = args.prompt_alpha if args.prompt_alpha != 1.0 else a1.get("prompt_alpha", 1.0)
    hb = args.hyb_bound if args.hyb_bound > 0.0 else a1.get("hyb_bound", 0.0)
    rb = args.rot_bound if args.rot_bound > 0.0 else float(a1.get("rot_bound", 0.0))
    a1["prompt_alpha"], a1["hyb_bound"], a1["rot_bound"] = pa, hb, rb
    vrmgs, registry = attach_rotation(model, layers, block_size=a1["block"], multifreq=a1.get("multifreq", False),
                                      tokrot=a1.get("tokrot", False), hybrid=a1.get("hybrid", False),
                                      visual_dim=data["vfeat"][0].shape[-1], query_dim=a1["query_dim"],
                                      prompt_alpha=pa, hyb_bound=hb,
                                      qcond=args.qcond or a1.get("qcond", False),
                                      direct_hyb=args.direct_hyb or a1.get("direct_hyb", False),
                                      no_rot=args.no_rot or a1.get("no_rot", False),
                                      rot_bound=rb)
    for m, sd in zip(vrmgs, ck["state"]):
        m.load_state_dict(sd)
    freeze_all_but(model, [])
    vparams = [p for m in vrmgs for p in m.parameters()]
    for p in vparams:
        p.requires_grad_(True)
    all_params = list(vparams)
    pre_hashes, pre_bb_sha1 = snapshot_backbone_hashes(model)

    teacher = None
    if not args.eval_only:
        teacher = AutoModelForImageTextToText.from_pretrained(args.teacher, dtype=torch.bfloat16, device_map=dev,
                                                              trust_remote_code=True).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    log(f"=== OPD steps={args.steps} bs={args.bs} max_new={args.max_new} min_new={args.min_new} lr={args.lr} "
        f"split={split_info} ce_w={args.ce_weight} train={len(train)} val={len(val)}(em_n={len(val_em_1000)}) test={len(tidx)} ===")

    def run_eval(tag, with_test=True, n_val_tf=None, n_val_em=None):
        out = {}
        ev = evaluate(model, registry, tok, data, val_em_1000, dev, n_val_tf or min(150, len(val_em_1000)),
                      n_val_em or len(val_em_1000), args.max_new, args.boxed)
        log(f"{tag} val: {fmt(ev)}")
        out["val"] = ev
        if with_test and tdata is not None:
            et = evaluate(model, registry, tok, tdata, tidx, dev, len(tidx), len(tidx), args.max_new, args.boxed)
            log(f"{tag} test: {fmt(et)}")
            out["test"] = et
        return out

    def save(res, manifest=None):
        if manifest is not None:
            res["manifest"] = manifest
        with open(args.out + ".json", "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        torch.save({"state": [m.state_dict() for m in vrmgs], "args": a1,
                    "manifest": manifest, "backbone": None}, args.out + ".pt")

    result = {"args": vars(args), "split_info": split_info, "val_em_sha1": val_1000_sha1, "mid_eval": []}
    init = run_eval("INIT")
    result["init_eval"] = init["val"]
    if "test" in init:
        result["init_test"] = init["test"]
    if args.eval_only:
        result["eval"], result["test"] = init["val"], init.get("test")
        save(result, build_run_manifest("tower_opd_eval", vars(args), split_info))
        log("OPD_DONE"); return

    opt = torch.optim.AdamW([{"params": vparams, "lr": args.lr}], weight_decay=0.01)
    verify_frozen_backbone_and_optimizer(model, opt, all_params)
    eval_every = args.eval_every or max(args.steps // 3, 100)
    t0, hist, gn = time.time(), [], torch.zeros(())
    pad_id = tok.pad_token_id
    for ms in range(1, args.steps * args.accum + 1):
        step, micro = (ms - 1) // args.accum + 1, (ms - 1) % args.accum
        if micro == 0:
            f = min(1.0, step / 50) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            for g in opt.param_groups:
                g["lr"] = args.lr * f
            opt.zero_grad(set_to_none=True)
        idx = [train[(ms * args.bs + k) % len(train)] for k in range(args.bs)]
        prompts = [data["prompts"][i] for i in idx]
        pids = [tok(p)["input_ids"] for p in prompts]
        pad = max(len(x) for x in pids)
        in_ids = torch.tensor([[pad_id] * (pad - len(x)) + x for x in pids], device=dev)
        attn = torch.tensor([[0] * (pad - len(x)) + [1] * len(x) for x in pids], device=dev)
        vis, vmask = pad_vis([data["vfeat"][i].float() for i in idx], dev)

        with torch.no_grad(), rotation_context_with(registry, vis, vmask,
                                                   ans_start=[pad] * len(pids), text_mask=attn):
            gen = model.generate(input_ids=in_ids, attention_mask=attn, max_new_tokens=args.max_new,
                                 min_new_tokens=args.min_new, do_sample=args.gen_temp > 0,
                                 temperature=max(args.gen_temp, 1e-5), top_p=0.9, pad_token_id=pad_id,
                                 eos_token_id=stops)
        gids = [cut_at_stop(gen[k, pad:], set(stops), include_stop=True) for k in range(len(idx))]

        imgs = [Image.open(paths[data["ids"][i]]).convert("RGB") for i in idx]
        enc, Lp, bad = build_teacher_batch(proc, prompts, pids, gids, imgs, dev, pad_id)
        with torch.no_grad():
            tlog = teacher(**enc).logits

        # 学生前向：布局与 generate 完全一致（左填充 prompt + 右填充答案），并显式给 position_ids。
        maxa = max([len(g) for g in gids] + [1])
        tail = torch.full((len(idx), maxa), pad_id, dtype=in_ids.dtype, device=dev)
        tmask = torch.zeros(len(idx), maxa, dtype=attn.dtype, device=dev)
        for k, g in enumerate(gids):
            if g:
                tail[k, : len(g)] = torch.tensor(g, device=dev)
                tmask[k, : len(g)] = 1
        seq, smask = torch.cat([in_ids, tail], dim=1), torch.cat([attn, tmask], dim=1)
        pos = (smask.cumsum(-1) - 1).clamp(min=0)
        with rotation_context_with(registry, vis, vmask, ans_start=[pad] * len(idx), text_mask=smask):
            s_lg = model(input_ids=seq, attention_mask=smask, position_ids=pos).logits[:, pad - 1 : pad - 1 + maxa]
        s_lg = s_lg.float()

        T = args.kd_temp
        kl_terms = []
        for k, g in enumerate(gids):
            if not g or bad[k]:
                continue
            na = len(g)
            t_logp = F.log_softmax(tlog[k, Lp - 1 : Lp - 1 + na].float() / T, dim=-1)
            s_logp = F.log_softmax(s_lg[k, :na] / T, dim=-1)
            kl_terms.append((t_logp.exp() * (t_logp - s_logp)).sum(-1).mean())
        loss = None
        if kl_terms:
            loss = torch.stack(kl_terms).mean() * T ** 2
        if args.ce_weight > 0:
            ids_g, mask_g, lab_g, vis_g, vm_g, *_ = tf.collate(idx, data, tok, args.seq, False)
            if bool((lab_g[:, 1:] != -100).any()):  # 整批标签都被截断时 CE 是 0/0=NaN，会把旋转参数写坏
                with rotation_context_with(registry, vis_g.to(dev), vm_g.to(dev),
                                           ans_start=[data["a_rel_start"][i] for i in idx], text_mask=mask_g.to(dev)):
                    lg_g = model(input_ids=ids_g.to(dev), attention_mask=mask_g.to(dev)).logits.float()
                ce = F.cross_entropy(lg_g[:, :-1].reshape(-1, lg_g.shape[-1]), lab_g.to(dev)[:, 1:].reshape(-1))
                loss = args.ce_weight * ce if loss is None else loss + args.ce_weight * ce
        if loss is not None and bool(torch.isfinite(loss)):
            (loss / args.accum).backward()
        hist.append((loss.item() if loss is not None else float("nan"), len(kl_terms),
                     sum(len(g) for g in gids) / len(gids), sum(bad)))
        if micro < args.accum - 1:
            continue
        gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        if step % args.log_every == 0:
            h = hist[-args.log_every:]
            log(f"step {step} loss={sum(x[0] for x in h)/len(h):.4f} kl_used={sum(x[1] for x in h)/len(h):.1f} "
                f"len={sum(x[2] for x in h)/len(h):.1f}tok bad_prefix={sum(x[3] for x in h)} "
                f"gn={float(gn):.2f} {(time.time()-t0)/step:.2f}s/it")
        if step % eval_every == 0 and step < args.steps:
            ev = run_eval(f"  mid@{step}", with_test=False, n_val_tf=60, n_val_em=40)
            result["mid_eval"].append({"step": step, **ev["val"]})

    frozen_info = verify_frozen_backbone_and_optimizer(model, opt, all_params, pre_hashes)
    result["train_seconds"] = time.time() - t0
    fin = run_eval("FINAL")
    result["eval"] = fin["val"]
    if "test" in fin:
        result["test"] = fin["test"]
    result["loss_tail"] = hist[-200:]
    result["frozen_check"] = frozen_info
    result["peak_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    manifest = build_run_manifest("tower_opd_train", vars(args), split_info,
                                  extra={"frozen_check": frozen_info, "val_em_sha1": val_1000_sha1,
                                         "backbone_sha1_before": pre_bb_sha1})
    save(result, manifest)
    for part, before in (("val", result["init_eval"]), ("test", result.get("init_test"))):
        after = fin.get(part)
        if before and after:
            log(f"DELTA {part}: " + " ".join(f"{k}:{before[k]:.4f}->{after[k]:.4f}"
                                              for k in ("tf_first", "tf_full", "em", "stop_rate")))
    log("OPD_DONE")


if __name__ == "__main__":
    main()
