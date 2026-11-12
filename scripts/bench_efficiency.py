"""效率与显存基准测试（独立生命周期隔离版）：
针对五种实际模型架构，在每个分辨率 N in {256, 576, 1024, 2304, 4096} 下，各自独占 GPU 测量：
  1. ours       : SpinVL-Free 最终架构 (PatchFront rots K=4 + LayerVRMG[mf, hyb, qcond, direct_hyb])
  2. tower_ours : SpinVL-Tower 最终架构 (External 0.41B ViT + LayerVRMG[mf, hyb, qcond, direct_hyb])
  3. rose       : ROSE 单频旋转基线 (External 0.41B ViT + LayerVRMG[single-freq, no hyb, no qcond])
  4. keep       : 保留视觉 Token 的内置编码器对照 (N/4 cells + Q tokens 跑满 L 层)
  5. llava      : LLaVA-Proj 全量视觉 Token 基线 (External 0.41B ViT + Projector + N/4 cells + Q tokens 跑满 L 层)

每个模型独立构建、独立统计显存峰值 (peak_gib)，并在同一趟运行内用 use_cache=True 连续打点测量：
  - prefill_ms : 视觉特征提取 + 文本 Prompt 首步前向（产出第 1 个新 token 与 KV cache）
  - decode_ms  : 基于 KV cache 连续贪婪解码剩余 (gen_tokens - 1) 个 token 的耗时
  - total_ms   : prefill_ms + decode_ms 端到端总耗时
"""

import argparse
import gc
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from front import PatchFront, unwrap_layer  # noqa: E402
from provenance import build_run_manifest  # noqa: E402
from wrap import attach_rotation, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
Q = 64


def grid_for(n):
    s = int(round(n ** 0.5))
    return torch.tensor([[1, s, s]]), s * s


def transformer_flops(d, ff, nl, L, q_dim, kv_dim, causal=True):
    """解析式前向 FLOPs（TFLOPs）：线性层乘加算 2，注意力分数与加权各一次，causal 取一半。"""
    lin = 2 * L * (d * q_dim + 2 * d * kv_dim + q_dim * d + 3 * d * ff)
    attn = 2 * 2 * (L * L / 2 if causal else L * L) * q_dim
    return nl * (lin + attn) / 1e12


def compute_analytic_tflops(model_cfg, vcfg, K, n_patch, seq_len, n_vrmg_layers=7):
    cells = n_patch // 4
    d, ff = model_cfg.hidden_size, model_cfg.intermediate_size
    q_dim = model_cfg.num_attention_heads * getattr(model_cfg, "head_dim", d // model_cfg.num_attention_heads)
    kv_dim = model_cfg.num_key_value_heads * getattr(model_cfg, "head_dim", d // model_cfg.num_attention_heads)
    vh, vff = vcfg.hidden_size, vcfg.intermediate_size
    vq = vh
    v_depth = getattr(vcfg, "depth", getattr(vcfg, "num_hidden_layers", 32))
    v_out = getattr(vcfg, "out_hidden_size", 4096)

    # 1. 外部视觉塔 FLOPs
    tower_pe = 2 * n_patch * 1536 * vh
    tower_body = transformer_flops(vh, vff, v_depth, n_patch, vq, vq, causal=False) * 1e12
    tower_mg = 2 * cells * (4 * vh) * v_out * 2
    tower_tf = (tower_pe + tower_body + tower_mg) / 1e12

    # 2. 内置前端 PatchFront (rots K=4) FLOPs
    front_pe = 2 * n_patch * 1536 * d
    front_rot = K * 2 * n_patch * (4 * d + 2 * kv_dim + 3 * ff) * 64 + K * n_patch * (4 * d + 2 * kv_dim + 3 * ff)
    front_body = transformer_flops(d, ff, K, n_patch, q_dim, kv_dim, causal=False) * 1e12
    front_mg = 2 * cells * (4 * d) * v_out * 2
    front_tf = (front_pe + front_rot + front_body + front_mg) / 1e12

    # 3. VRMG 算力：区分单频 ROSE 与完整 SpinVL (mf + qcond + direct_hyb)
    total_blocks = (4 * d + 2 * kv_dim + 3 * ff) // 64
    qdim_v = 256
    kv_proj = 2 * 2 * cells * v_out * qdim_v
    q_attn = 2 * 2 * total_blocks * cells * qdim_v
    out_lin = 2 * total_blocks * qdim_v * (64 * 63 // 2)
    cayley = total_blocks * (2 * (64 ** 3))
    apply_rot = 2 * seq_len * (4 * d + 2 * kv_dim + 3 * ff) * 64
    vrmg_single_tf = n_vrmg_layers * (kv_proj + q_attn + out_lin + cayley + apply_rot) / 1e12

    qcond_ops = 2 * d * qdim_v + seq_len * d
    gate_ops = 2 * seq_len * d * total_blocks
    hyb_ops = 2 * 2 * cells * v_out * qdim_v + 2 * seq_len * d * qdim_v * 2 + 2 * 2 * seq_len * cells * qdim_v
    vrmg_full_tf = vrmg_single_tf + n_vrmg_layers * (qcond_ops + gate_ops + hyb_ops) / 1e12

    llm_q_tf = transformer_flops(d, ff, model_cfg.num_hidden_layers, seq_len, q_dim, kv_dim)
    llm_full_tf = transformer_flops(d, ff, model_cfg.num_hidden_layers, cells + seq_len, q_dim, kv_dim)
    proj_tf = (2 * cells * v_out * d * 2) / 1e12

    return {
        "ours": front_tf + llm_q_tf + vrmg_full_tf,
        "tower_ours": tower_tf + llm_q_tf + vrmg_full_tf,
        "rose": tower_tf + llm_q_tf + vrmg_single_tf,
        "keep": llm_full_tf,
        "llava": tower_tf + proj_tf + llm_full_tf,
    }


def unwrap_all_layers(model):
    for i in range(len(model.model.layers)):
        model.model.layers[i] = unwrap_layer(model.model.layers[i])


def run_prefill_and_decode(prefill_fn, step_fn, gen_tokens: int, warm: int = 2, runs: int = 8):
    """在同一趟内连续测量 prefill_ms 与 decode_ms (gen_tokens - 1 步)，保证 prefill_ms + decode_ms == total_ms。"""
    def one_trial():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        next_tok, past_kv, ctx_Exit = prefill_fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        try:
            cur_tok = next_tok
            for _ in range(max(0, gen_tokens - 1)):
                cur_tok, past_kv = step_fn(cur_tok, past_kv)
        finally:
            if ctx_Exit is not None:
                ctx_Exit.__exit__(None, None, None)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        return (t1 - t0) * 1000.0, (t2 - t1) * 1000.0, (t2 - t0) * 1000.0

    for _ in range(warm):
        one_trial()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pf_list, dc_list, tot_list = [], [], []
    for _ in range(runs):
        pf, dc, tot = one_trial()
        pf_list.append(pf)
        dc_list.append(dc)
        tot_list.append(tot)
    peak_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    return {
        "ms": statistics.median(pf_list),
        "prefill_ms": statistics.median(pf_list),
        "prefill_mean_ms": statistics.mean(pf_list),
        "prefill_std_ms": statistics.stdev(pf_list) if len(pf_list) > 1 else 0.0,
        "decode_ms": statistics.median(dc_list),
        "decode_mean_ms": statistics.mean(dc_list),
        "decode_std_ms": statistics.stdev(dc_list) if len(dc_list) > 1 else 0.0,
        "total_ms": statistics.median(tot_list),
        "total_mean_ms": statistics.mean(tot_list),
        "total_std_ms": statistics.stdev(tot_list) if len(tot_list) > 1 else 0.0,
        "peak_gib": peak_gib,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--Ns", default="256,576,1024,2304,4096")
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--joint-ckpt", default="", help="可选：SpinVL-Free 联合训练/OPD 权重 (.pt)")
    ap.add_argument("--tower-ckpt", default="", help="可选：SpinVL-Tower 权重 (.pt)")
    ap.add_argument("--rose-ckpt", default="", help="可选：ROSE 权重 (.pt)")
    ap.add_argument("--front-ckpt", default="", help="向后兼容：仅加载前端权重")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    vcfg = AutoConfig.from_pretrained(f"{B}/zk/model/Qwen3-VL-8B-Instruct", trust_remote_code=True).vision_config
    model_path = f"{B}/zk/model/Qwen3-1.7B"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = [0, 4, 8, 12, 16, 20, 24]
    Ns = [int(x) for x in args.Ns.split(",")]
    rows_by_n = {}
    for N in Ns:
        _, n_patch = grid_for(N)
        tf_est = compute_analytic_tflops(model.config, vcfg, args.K, n_patch, Q, len(layers))
        rows_by_n[n_patch] = {
            "N_patches": n_patch,
            "cells": n_patch // 4,
            "gen_tokens": args.gen_tokens,
            "_tf_est": tf_est,
        }

    txt = torch.randint(100, 1000, (1, Q), device=dev)
    attn_txt = torch.ones_like(txt)

    # 逐模型独立构建、测试与释放，避免不同架构共用 registry 或污染显存峰值
    for mode_name in ("ours", "tower_ours", "rose", "keep", "llava"):
        unwrap_all_layers(model)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        vrmgs, registry, front, tower, proj = [], {}, None, None, None
        if mode_name == "ours":
            vrmgs, registry = attach_rotation(
                model, layers, block_size=64, visual_dim=4096, query_dim=256,
                wrap_extra=[l for l in range(args.K) if l not in layers],
                multifreq=True, hybrid=True, qcond=True, direct_hyb=True
            )
            front = PatchFront(model, K=args.K, mode="rots", vdim=4096).to(dev).eval()
            if args.joint_ckpt and os.path.exists(args.joint_ckpt):
                jck = torch.load(args.joint_ckpt, map_location=dev)
                front.load_state_dict(jck["front"])
                for m, sd in zip(vrmgs, jck["vrmg"]):
                    m.load_state_dict(sd)
            elif args.front_ckpt and os.path.exists(args.front_ckpt):
                front.load_state_dict(torch.load(args.front_ckpt, map_location=dev)["front"])
            for m in vrmgs:
                m.eval()
        elif mode_name == "tower_ours":
            vrmgs, registry = attach_rotation(
                model, layers, block_size=64, visual_dim=4096, query_dim=256,
                multifreq=True, hybrid=True, qcond=True, direct_hyb=True
            )
            if args.tower_ckpt and os.path.exists(args.tower_ckpt):
                tck = torch.load(args.tower_ckpt, map_location=dev)
                for m, sd in zip(vrmgs, tck["state"]):
                    m.load_state_dict(sd)
            for m in vrmgs:
                m.eval()
            tower = Qwen3VLVisionModel(vcfg).to(dev, dtype=torch.bfloat16).eval()
        elif mode_name == "rose":
            # 严格使用单频、无 hybrid、无 qcond 的 ROSE 基线结构
            vrmgs, registry = attach_rotation(
                model, layers, block_size=64, visual_dim=4096, query_dim=256,
                multifreq=False, hybrid=False, qcond=False, direct_hyb=False
            )
            if args.rose_ckpt and os.path.exists(args.rose_ckpt):
                rck = torch.load(args.rose_ckpt, map_location=dev)
                for m, sd in zip(vrmgs, rck["state"]):
                    m.load_state_dict(sd)
            for m in vrmgs:
                m.eval()
            tower = Qwen3VLVisionModel(vcfg).to(dev, dtype=torch.bfloat16).eval()
        elif mode_name == "llava":
            tower = Qwen3VLVisionModel(vcfg).to(dev, dtype=torch.bfloat16).eval()
            proj = torch.nn.Sequential(
                torch.nn.Linear(4096, model.config.hidden_size),
                torch.nn.GELU(),
                torch.nn.Linear(model.config.hidden_size, model.config.hidden_size),
            ).to(dev, dtype=torch.bfloat16).eval()

        emb = model.get_input_embeddings()
        with torch.no_grad():
            for N in Ns:
                grid, n_patch = grid_for(N)
                cells = n_patch // 4
                pv = torch.randn(1, n_patch, 1536, device=dev, dtype=torch.float16)
                tower_in = torch.randn(n_patch, 1536, device=dev, dtype=torch.bfloat16)
                vis_tok = torch.randn(1, cells, model.config.hidden_size, device=dev, dtype=torch.bfloat16)
                vm = torch.ones(1, cells, device=dev)

                def make_fns(m_name):
                    if m_name == "ours":
                        def pf():
                            vh, cl = front(pv, grid, [n_patch])
                            ctx = rotation_context_with(registry, vh, vm[:, : cl[0]], text_mask=attn_txt)
                            ctx.__enter__()
                            o = model(input_ids=txt, attention_mask=attn_txt, use_cache=True)
                            nt = o.logits[:, -1:, :].argmax(dim=-1)
                            return nt, o.past_key_values, ctx

                        def st(cur_tok, pkv):
                            o = model(input_ids=cur_tok, past_key_values=pkv, use_cache=True)
                            return o.logits[:, -1:, :].argmax(dim=-1), o.past_key_values
                        return pf, st

                    if m_name in ("tower_ours", "rose"):
                        def pf():
                            vo = tower(tower_in, grid_thw=grid)
                            vf = (vo[0] if isinstance(vo, (tuple, list)) else vo).float().unsqueeze(0)
                            ctx = rotation_context_with(registry, vf, vm, text_mask=attn_txt)
                            ctx.__enter__()
                            o = model(input_ids=txt, attention_mask=attn_txt, use_cache=True)
                            nt = o.logits[:, -1:, :].argmax(dim=-1)
                            return nt, o.past_key_values, ctx

                        def st(cur_tok, pkv):
                            o = model(input_ids=cur_tok, past_key_values=pkv, use_cache=True)
                            return o.logits[:, -1:, :].argmax(dim=-1), o.past_key_values
                        return pf, st

                    if m_name == "keep":
                        def pf():
                            e = torch.cat([vis_tok, emb(txt)], dim=1)
                            o = model(inputs_embeds=e, use_cache=True)
                            nt = o.logits[:, -1:, :].argmax(dim=-1)
                            return nt, o.past_key_values, None

                        def st(cur_tok, pkv):
                            o = model(input_ids=cur_tok, past_key_values=pkv, use_cache=True)
                            return o.logits[:, -1:, :].argmax(dim=-1), o.past_key_values
                        return pf, st

                    def pf():
                        vo = tower(tower_in, grid_thw=grid)
                        vf = (vo[0] if isinstance(vo, (tuple, list)) else vo).to(torch.bfloat16).unsqueeze(0)
                        vt = proj(vf)
                        e = torch.cat([vt, emb(txt)], dim=1)
                        o = model(inputs_embeds=e, use_cache=True)
                        nt = o.logits[:, -1:, :].argmax(dim=-1)
                        return nt, o.past_key_values, None

                    def st(cur_tok, pkv):
                        o = model(input_ids=cur_tok, past_key_values=pkv, use_cache=True)
                        return o.logits[:, -1:, :].argmax(dim=-1), o.past_key_values
                    return pf, st

                pf_fn, st_fn = make_fns(mode_name)
                try:
                    stats = run_prefill_and_decode(pf_fn, st_fn, args.gen_tokens)
                    tflops_val = rows_by_n[n_patch]["_tf_est"][mode_name]
                    rows_by_n[n_patch][mode_name] = {
                        "tflops": tflops_val,
                        "est_prefill_tflops": tflops_val,
                        **stats,
                    }
                except Exception as e:
                    tb = traceback.extract_tb(e.__traceback__)[-1]
                    rows_by_n[n_patch][mode_name] = {
                        "error": f"{type(e).__name__}: {str(e)[:160]} @ {os.path.basename(tb.filename)}:{tb.lineno}"
                    }

        del vrmgs, registry, front, tower, proj
        gc.collect()
        torch.cuda.empty_cache()

    rows = []
    for N in Ns:
        _, n_patch = grid_for(N)
        r = rows_by_n[n_patch]
        r.pop("_tf_est", None)
        r["peak_gib"] = r.get("ours", {}).get("peak_gib", 0.0)
        print(json.dumps(r), flush=True)
        rows.append(r)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    manifest = build_run_manifest("efficiency", vars(args))
    json.dump({
        "K": args.K,
        "Q": Q,
        "gen_tokens": args.gen_tokens,
        "manifest": manifest,
        "rows": rows,
        "note": "各模型独立生命周期独占 GPU 测量；rose 为单频无 cross-attention 基线，ours/tower_ours 为完整架构；prefill_ms 与 decode_ms 为同趟连续打点",
    }, open(args.out, "w"), indent=1, ensure_ascii=False)
    print("BENCH_DONE")


if __name__ == "__main__":
    main()
