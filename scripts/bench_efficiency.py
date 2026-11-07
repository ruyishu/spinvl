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


def timed(fn, warm=2, runs=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        t = time.time(); fn(); torch.cuda.synchronize(); ts.append((time.time() - t) * 1000)
    return statistics.median(ts)


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
    ap.add_argument("--front-ckpt", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(f"{B}/zk/model/Qwen3-1.7B", dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    layers = [0, 4, 8, 12, 16, 20, 24]
    vrmgs, registry = attach_rotation(model, layers, block_size=64, visual_dim=4096, query_dim=256,
                                      wrap_extra=[l for l in range(args.K) if l not in layers])
    front = PatchFront(model, K=args.K, mode="rot", vdim=4096).to(dev)
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

            def ours():
                vh, cl = front(pv, grid, [n_patch])
                with rotation_context_with(registry, vh, vm[:, : cl[0]]):
                    return model(input_ids=txt)

            def rose():
                tower(tower_in, grid_thw=grid)
                with rotation_context_with(registry, v_hat, vm):
                    return model(input_ids=txt)

            def keep():
                e = torch.cat([vis_tok, emb(txt)], dim=1)
                return model(inputs_embeds=e)

            def llava():
                tower(tower_in, grid_thw=grid)
                e = torch.cat([vis_tok, emb(txt)], dim=1)
                return model(inputs_embeds=e)

            row = {"N_patches": n_patch, "cells": cells}
            # 解析式 FLOPs（不依赖 PyTorch FlopCounterMode，2.8 下会触发内部断言）
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
                body = transformer_flops(d, ff, args.K, n, q_dim, kv_dim, causal=False) * 1e12
                mg = 2 * (n // 4) * (4 * d) * 4096 * 2
                return (pe + body + mg) / 1e12

            tf_map = {
                "ours": front_flops(n_patch) + transformer_flops(d, ff, cfg.num_hidden_layers, Q, q_dim, kv_dim),
                "rose": tower_flops(n_patch) + transformer_flops(d, ff, cfg.num_hidden_layers, Q, q_dim, kv_dim),
                "keep": transformer_flops(d, ff, cfg.num_hidden_layers, cells + Q, q_dim, kv_dim),
                "llava": tower_flops(n_patch) + transformer_flops(d, ff, cfg.num_hidden_layers, cells + Q, q_dim, kv_dim),
            }
            for name, fn in (("ours", ours), ("rose", rose), ("keep", keep), ("llava", llava)):
                try:
                    row[name] = {"tflops": tf_map[name], "ms": timed(fn)}
                except Exception as e:  # 极端长度可能显存不够，记录而不中断；类型+末帧位置必须记，空消息异常否则看不出原因
                    tb = traceback.extract_tb(e.__traceback__)[-1]
                    row[name] = {"error": f"{type(e).__name__}: {str(e)[:160]} @ {os.path.basename(tb.filename)}:{tb.lineno}"}
                    print(f"[bench] {name} N={n_patch} failed: {row[name]['error']}", flush=True)
            row["peak_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            print(json.dumps(row), flush=True)
            rows.append(row)
            torch.cuda.reset_peak_memory_stats()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"K": args.K, "Q": Q, "rows": rows,
               "note": "latency 仅在独占 GPU 时可信；keep 不计 K 层读图的额外开销（偏向对手）"},
              open(args.out, "w"), indent=1)
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
