"""效率对比：单图推理 prefill 的 FLOPs 与延迟，随视觉 patch 数 N 变化。

四种结构（学生均为 Qwen3-1.7B，文本问题 Q=64 token）：
  ours      ：内置前端读图（前 K 层吃 N 个 patch）→ 丢弃图片 token → 后续只跑 Q 个文本 token（+VRMG 动态旋转）
  keep      ：encoder-free 且图片 token 全程保留（VoRA/EVE 式）：N/4 个 cell + Q 跑满 28 层（含 K 层读图的同等开销不计，偏向对手）
  llava     ：外挂视觉塔（随机权重，只测开销）+ N/4 个视觉 token + Q 跑满 28 层
  rose      ：外挂视觉塔 + VRMG 旋转（文本通路无视觉 token）
FLOPs 用 torch.utils.flop_counter（与是否有并发任务无关）；延迟需独占 GPU 才可信，结果里记录 GPU 占用情况。
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from front import PatchFront  # noqa: E402
from wrap import attach_rotation, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
Q = 64


def grid_for(n):
    s = int(round(n ** 0.5))
    return torch.tensor([[1, s, s]]), s * s


def timed_stats(fn, warm=2, runs=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        t = time.time(); fn(); torch.cuda.synchronize(); ts.append((time.time() - t) * 1000)
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    return {
        "median_ms": statistics.median(ts),
        "mean_ms": statistics.mean(ts),
        "std_ms": statistics.stdev(ts) if len(ts) > 1 else 0.0,
        "peak_gib": peak_gib,
    }


def timed(fn, warm=2, runs=8):
    return timed_stats(fn, warm=warm, runs=runs)["median_ms"]


def flops_of(fn):
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops() / 1e12  # TFLOPs


def transformer_flops(d, ff, nl, L, q_dim, kv_dim, causal=True):
    """解析式前向 FLOPs（TFLOPs）：线性层乘加算 2，注意力分数与加权各一次，causal 取一半。"""
    lin = 2 * L * (d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * ff)
    attn = 2 * 2 * (L * L / 2 if causal else L * L) * q_dim
    return nl * (lin + attn) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--Ns", default="256,576,1024,2304,4096")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--front-ckpt", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    layers = [0, 4, 8, 12, 16, 20, 24]
    vrmgs, registry = attach_rotation(model, layers, block_size=64, visual_dim=4096, query_dim=256,
                                      wrap_extra=[l for l in range(args.K) if l not in layers],
                                      multifreq=True, hybrid=True, qcond=True, direct_hyb=True)
    front = PatchFront(model, K=args.K, mode="rots", vdim=4096).to(dev)
    if args.front_ckpt:
        front.load_state_dict(torch.load(args.front_ckpt, map_location=dev)["front"])
    front.eval()

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    vcfg = AutoConfig.from_pretrained(f"{B}/zk/model/Qwen3-VL-8B-Instruct", trust_remote_code=True).vision_config
    tower = Qwen3VLVisionModel(vcfg).to(dev, dtype=torch.bfloat16).eval()
    emb = model.get_input_embeddings()
    txt = torch.randint(100, 1000, (1, Q), device=dev)
    rows = []
    with torch.no_grad():
        for N in [int(x) for x in args.Ns.split(",")]:
            grid, n_patch = grid_for(N)
            pv = torch.randn(1, n_patch, 1536, device=dev, dtype=torch.float16)
            cells = n_patch // 4
            v_hat = torch.randn(1, cells, 4096, device=dev)
            vm = torch.ones(1, cells, device=dev)
            tower_in = torch.randn(n_patch, 1536, device=dev, dtype=torch.bfloat16)
            vis_tok = torch.randn(1, cells, model.config.hidden_size, device=dev, dtype=torch.bfloat16)

            attn_txt = torch.ones_like(txt)

            def ours():
                vh, cl = front(pv, grid, [n_patch])
                with rotation_context_with(registry, vh, vm[:, : cl[0]], text_mask=attn_txt):
                    return model(input_ids=txt, attention_mask=attn_txt)

            def ours_e2e():
                vh, cl = front(pv, grid, [n_patch])
                with rotation_context_with(registry, vh, vm[:, : cl[0]], text_mask=attn_txt):
                    return model.generate(input_ids=txt, attention_mask=attn_txt,
                                          max_new_tokens=args.gen_tokens, min_new_tokens=args.gen_tokens,
                                          do_sample=False, pad_token_id=151643)

            def rose():
                tower(tower_in, grid_thw=grid)
                with rotation_context_with(registry, v_hat, vm, text_mask=attn_txt):
                    return model(input_ids=txt, attention_mask=attn_txt)

            def rose_e2e():
                tower(tower_in, grid_thw=grid)
                with rotation_context_with(registry, v_hat, vm, text_mask=attn_txt):
                    return model.generate(input_ids=txt, attention_mask=attn_txt,
                                          max_new_tokens=args.gen_tokens, min_new_tokens=args.gen_tokens,
                                          do_sample=False, pad_token_id=151643)

            def keep():
                e = torch.cat([vis_tok, emb(txt)], dim=1)
                return model(inputs_embeds=e)

            def llava():
                tower(tower_in, grid_thw=grid)
                e = torch.cat([vis_tok, emb(txt)], dim=1)
                return model(inputs_embeds=e)

            def llava_e2e():
                tower(tower_in, grid_thw=grid)
                e = torch.cat([vis_tok, emb(txt)], dim=1)
                m = torch.ones(1, e.shape[1], device=dev, dtype=torch.long)
                return model.generate(inputs_embeds=e, attention_mask=m,
                                      max_new_tokens=args.gen_tokens, min_new_tokens=args.gen_tokens,
                                      do_sample=False, pad_token_id=151643)

            row = {"N_patches": n_patch, "cells": cells, "gen_tokens": args.gen_tokens}
            # 解析式估算 FLOPs（含 Transformer 主干、视觉塔/PatchFront、以及 7 层 VRMG 旋转生成+Cayley+块乘法+direct_hyb 交叉注意力）
            cfg = model.config
            d, ff = cfg.hidden_size, cfg.intermediate_size
            q_dim = cfg.num_attention_heads * getattr(cfg, "head_dim", d // cfg.num_attention_heads)
            kv_dim = cfg.num_key_value_heads * getattr(cfg, "head_dim", d // cfg.num_attention_heads)
            vh, vff = vcfg.hidden_size, vcfg.intermediate_size
            vq = vh  # 视觉塔 MHA 无 GQA

            def tower_flops(n):
                pe = 2 * n * 1536 * vh
                body = transformer_flops(vh, vff, getattr(vcfg, 'depth', getattr(vcfg, 'num_hidden_layers', 32)), n, vq, vq, causal=False) * 1e12
                mg = 2 * (n // 4) * (4 * vh) * getattr(vcfg, "out_hidden_size", 4096) * 2
                return (pe + body + mg) / 1e12

            def front_flops(n):
                pe = 2 * n * 1536 * d
                # StaticRotScale 在前 K 层每步额外对输入做 64x64 块正交乘法
                rot_in = args.K * 2 * n * (4 * d + 3 * ff) * 64
                body = transformer_flops(d, ff, args.K, n, q_dim, kv_dim, causal=False) * 1e12
                mg = 2 * (n // 4) * (4 * d) * 4096 * 2
                return (pe + rot_in + body + mg) / 1e12

            def vrmg_flops(n_cells, seq_len, include_hyb=True):
                n_vrmg = len(layers)
                total_blocks = (4 * d + 2 * kv_dim + 3 * ff) // 64
                qdim_v = 256
                # 1. k/v 投影 + query cross-attention + out 线性层 + Cayley 64x64 求逆
                kv_proj = 2 * 2 * n_cells * 4096 * qdim_v
                q_attn = 2 * 2 * total_blocks * n_cells * qdim_v
                out_lin = 2 * total_blocks * qdim_v * (64 * 63 // 2)
                cayley = total_blocks * (2 * (64 ** 3))
                # 2. 文本序列在 7 个旋转投影上的 64x64 块乘法
                apply_rot = 2 * seq_len * (4 * d + 2 * kv_dim + 3 * ff) * 64
                # 3. direct_hyb 256 维窄子空间视觉交叉注意力
                hyb = 0
                if include_hyb:
                    hyb = 2 * 2 * n_cells * 4096 * qdim_v + 2 * seq_len * d * qdim_v * 2 + 2 * 2 * seq_len * n_cells * qdim_v
                return n_vrmg * (kv_proj + q_attn + out_lin + cayley + apply_rot + hyb) / 1e12

            tf_map = {
                "ours": front_flops(n_patch) + transformer_flops(d, ff, cfg.num_hidden_layers, Q, q_dim, kv_dim) + vrmg_flops(cells, Q, True),
                "rose": tower_flops(n_patch) + transformer_flops(d, ff, cfg.num_hidden_layers, Q, q_dim, kv_dim) + vrmg_flops(cells, Q, False),
                "keep": transformer_flops(d, ff, cfg.num_hidden_layers, cells + Q, q_dim, kv_dim),
                "llava": tower_flops(n_patch) + transformer_flops(d, ff, cfg.num_hidden_layers, cells + Q, q_dim, kv_dim),
            }
            e2e_map = {"ours": ours_e2e, "rose": rose_e2e, "llava": llava_e2e}
            for name, fn in (("ours", ours), ("rose", rose), ("keep", keep), ("llava", llava)):
                try:
                    pf_st = timed_stats(fn)
                    entry = {
                        "tflops": tf_map[name],
                        "est_prefill_tflops": tf_map[name],
                        "ms": pf_st["median_ms"],
                        "prefill_ms": pf_st["median_ms"],
                        "prefill_std_ms": pf_st["std_ms"],
                        "peak_gib": pf_st["peak_gib"],
                    }
                    if name in e2e_map:
                        e2e_st = timed_stats(e2e_map[name], warm=1, runs=5)
                        entry["total_ms"] = e2e_st["median_ms"]
                        entry["decode_ms"] = max(0.0, e2e_st["median_ms"] - pf_st["median_ms"])
                        entry["e2e_peak_gib"] = e2e_st["peak_gib"]
                    row[name] = entry
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[-1]
                    row[name] = {"error": f"{type(e).__name__}: {str(e)[:160]} @ {os.path.basename(tb.filename)}:{tb.lineno}"}
                    print(f"[bench] {name} N={n_patch} failed: {row[name]['error']}", flush=True)
            row["peak_gib"] = max((row[k].get("peak_gib", 0.0) for k in ("ours", "rose", "keep", "llava") if isinstance(row.get(k), dict)), default=0.0)
            print(json.dumps(row), flush=True)
            rows.append(row)
            torch.cuda.reset_peak_memory_stats()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"K": args.K, "Q": Q, "gen_tokens": args.gen_tokens,
               "arch": {"front_mode": "rots", "multifreq": True, "hybrid": True, "qcond": True, "direct_hyb": True},
               "rows": rows,
               "note": "est_prefill_tflops 为包含 VRMG/Cayley/direct_hyb 的解析式估算；prefill_ms/decode_ms/total_ms 为独占 GPU 实测"},
              open(args.out, "w"), indent=1)
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
