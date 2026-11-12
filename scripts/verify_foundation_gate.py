"""实验底座门禁验收脚本（完整覆盖方案第 2 节全部 7 项必修要求及二轮复核补强项）：
1. 统一生成与验证生成协议检查（取消按榜单首 token 禁词；Tower/Free/OPD 验证生成统一 max_new=24, bs=8, n=1000；禁用 --tune-backbone）
2. 修 ChartQA 计分检查（无隐式除以 100、无百分比二次除以 100、无任意包含匹配、统一抽取函数）
3. 修内部异图对照检查（val 64 / 600 / 4441 异图置换 0 碰撞；build_inbatch_diff_image_perm 连续 500 批及极端同图批 0 同图负例碰撞；公开榜 3 次独立随机图像组异图置换）
4. 统一跨阶段数据划分入口检查（直接调用 train_front.prepare_front_split、train_joint.prepare_joint_split、train_final.split_train_val_by_image 核对跨阶段 train/val 样本集与图像 SHA1 集 100% 一致且零重叠）
5. 验证 A1 (CA-only, no_rot=True) 与 A3+B2 冻结及梯度检查（A1 零旋转参数、cross-attention 梯度非零、更新后有图前向改变输出、无图前向逐位一致、训练前后 311 个原始 LLM 张量 SHA1 零变化）
6. 验证文本旁路检查（无图前向、先看图再切无图、走完 PatchFront 再切无图均与原装模型 logits 逐位一致）
7. 固定结果来源（调用 provenance.build_run_manifest 写出 results/clean_v6/foundation_gate_manifest.json）
"""

import inspect
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import bench_data as bd  # noqa: E402
import bench_models as bm  # noqa: E402
import bench_score as bs  # noqa: E402
import eval_benchmarks as eb  # noqa: E402
import provenance as prov  # noqa: E402
import train_final as tf  # noqa: E402
import train_front as tfront  # noqa: E402
import train_joint as tj  # noqa: E402
import train_joint_opd as tjopd  # noqa: E402
import train_opd as topd  # noqa: E402
from front import PatchFront  # noqa: E402
from wrap import attach_rotation, freeze_all_but, rotation_context_with  # noqa: E402

B = "/data/ibu-bi-ai-share-disk"
PROJ_DIR = Path(f"{B}/zk/project/rose-distill")
OUT_DIR = PROJ_DIR / "results" / "clean_v6"


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}

    # 1. 统一生成协议、统一验证生成协议与禁用 --tune-backbone 检查
    bm_src = (ROOT / "scripts" / "bench_models.py").read_text(encoding="utf-8")
    assert "FirstStepBanLetterProcessor" not in bm_src, "bench_models.py 仍残留 FirstStepBanLetterProcessor！"
    assert "logits_processor" not in bm_src, "bench_models.py 仍残留按榜单定制的 logits_processor！"

    sig_tf = inspect.signature(tf.eval_greedy_stats)
    sig_tj = inspect.signature(tj.front_greedy_em)
    sig_topd = inspect.signature(topd.greedy_em)
    assert sig_tf.parameters["max_new"].default == 24 and sig_tf.parameters["n"].default == 1000
    assert sig_tj.parameters["max_new"].default == 24 and sig_tj.parameters["n"].default == 1000
    assert sig_topd.parameters["max_new"].default == 24 and sig_topd.parameters["n"].default == 1000
    assert tjopd.front_greedy_em is tj.front_greedy_em

    opd_src = (ROOT / "scripts" / "train_opd.py").read_text(encoding="utf-8")
    jopd_src = (ROOT / "scripts" / "train_joint_opd.py").read_text(encoding="utf-8")
    assert 'raise ValueError("当前冻结主线严禁启用 --tune-backbone' in opd_src
    assert 'raise ValueError("当前冻结主线严禁启用 --tune-backbone' in jopd_src

    report["checks"]["1_unified_generation_and_val_protocol"] = {
        "passed": True,
        "no_first_step_ban": True,
        "no_custom_logits_processor": True,
        "val_greedy_max_new": 24,
        "val_greedy_n": 1000,
        "tune_backbone_rejected": True,
    }
    print("[PASS] 1. Unified benchmark & validation generation protocol + tune_backbone rejection verified.")

    # 2. ChartQA 计分自检（含百分比二次换算反例）
    bs.selftest()
    assert not bs.hit_relaxed("100", ["1"])
    assert not bs.hit_relaxed("10", ["0.1"])
    assert not bs.hit_relaxed("24%", ["0.0024"])
    assert not bs.hit_relaxed("100%", ["0.01"])
    assert not bs.hit_relaxed("0.24", ["0.0024%"])
    assert not bs.hit_relaxed("The answer is 24%", ["0.0024"])
    assert not bs.hit_relaxed("The. The. No. The", ["No"])
    assert bs.hit_relaxed("24%", ["0.24"]) and bs.hit_relaxed("24%", ["24"])
    assert bs.hit_relaxed("24", ["24%"]) and bs.hit_relaxed("0.24", ["24%"])
    report["checks"]["2_chartqa_scorer"] = {
        "passed": True,
        "scorer_version": bs.SCORER_VERSION,
        "rejects_implicit_div100": True,
        "rejects_double_pct_div10000": True,
        "rejects_substring_no": True,
    }
    print("[PASS] 2. ChartQA official relaxed scorer (including double-% counterexamples) verified.")

    # 3 & 4. 跨阶段实际入口数据切分一致性 + 内部/批内/公开榜异图置换检查
    data_spec = (
        f"{PROJ_DIR}/data/mix_v2/shard*.meta.pt,"
        f"{PROJ_DIR}/data/open_supp_v2/shard*.meta.pt,"
        f"{PROJ_DIR}/data/nat_supp/shard*.meta.pt"
    )
    sources = [
        f"{PROJ_DIR}/data/mix_v2/shard*.meta.pt|{PROJ_DIR}/data/mix_v2/manifest.jsonl|{PROJ_DIR}/data/mix",
        f"{PROJ_DIR}/data/open_supp_v2/shard*.meta.pt|{PROJ_DIR}/data/open_supp_v2/manifest.jsonl|{PROJ_DIR}/data/open_supp",
        f"{PROJ_DIR}/data/nat_supp/shard*.meta.pt|{PROJ_DIR}/data/nat_supp/manifest.jsonl|{PROJ_DIR}/data/nat_supp",
    ]
    gate_args = {"data": data_spec, "source": sources, "model": f"{B}/zk/model/Qwen3-1.7B"}
    prov.capture_run_inputs(gate_args)
    # 直接调用 train_front 与 train_joint 的真实入口函数
    data_f, paths_f, tr_f0, va_f0, tr_items_f0, va_items_f0, info_f0 = tfront.prepare_front_split(sources, seed=0)
    data_j, paths_j, tr_j0, va_j0, info_j0 = tj.prepare_joint_split(data_spec, sources, seed=0)
    tr_t0, va_t0, info_t0 = tf.split_train_val_by_image(data_j, None, seed=0)

    assert set(tr_f0) == set(tr_j0) == set(tr_t0), "train_front / train_joint / train_final 训练索引集不一致！"
    assert list(va_f0) == list(va_j0) == list(va_t0), "train_front / train_joint / train_final 验证索引序列不一致！"
    assert [x[0] for x in tr_items_f0] == tr_j0, "train_front 过滤后训练样本与 train_joint 不一致！"
    assert [x[0] for x in va_items_f0] == va_j0, "train_front 过滤后验证样本与 train_joint 不一致！"

    # 跨 seed (0, 1, 2) 核对
    for sd in (1, 2):
        _, _, tr_s, va_s, _ = tj.prepare_joint_split(data_spec, sources, seed=sd)
        assert set(tr_s) == set(tr_j0) and list(va_s) == list(va_j0)

    val = va_j0
    train = tr_j0
    split_info = info_j0
    assert split_info["group_overlap"] == 0 and split_info["train_rows"] == 39949 and split_info["val_rows"] == 4441

    val_1000, val_1000_sha1 = tf.select_stratified_val_1000(val, data_j, n_target=1000, seed=0)
    assert len(val_1000) == 1000 and len(set(val_1000)) == 1000

    # 3a. 验证内部异图置换在 64 / 600 / 4441 上均为 0 同图碰撞
    internal_derangements = {}
    for n_sub in (64, 600, len(val)):
        sub = val[:n_sub]
        shuf_sub, same_cnt, p_sha = tf.build_index_derangement(sub, data_j, paths_j, seed=20261112)
        assert same_cnt == 0, f"val[:{n_sub}] 存在同图碰撞 {same_cnt}"
        internal_derangements[str(n_sub)] = {"same_image_pairs": same_cnt, "perm_sha1": p_sha}

    # 3b. 验证训练批内负例函数 build_inbatch_diff_image_perm
    inbatch_checked_batches = 0
    inbatch_valid_pairs = 0
    for b_idx in range(500):
        batch = train[b_idx * 8 : (b_idx + 1) * 8]
        perm_l, valid_l = tf.build_inbatch_diff_image_perm(batch, data_j, paths_j)
        for j, (p_j, v_j) in enumerate(zip(perm_l, valid_l)):
            if v_j > 0:
                k1 = tf.get_image_key(batch[j], data_j, paths_j)
                k2 = tf.get_image_key(batch[p_j], data_j, paths_j)
                assert k1 != k2, f"批内负例碰撞: batch={b_idx} j={j}"
                inbatch_valid_pairs += 1
        inbatch_checked_batches += 1
    # 极端批测试：全同图批应全部返回 valid=0.0
    same_batch = [train[0]] * 8
    _, same_valid = tf.build_inbatch_diff_image_perm(same_batch, data_j, paths_j)
    assert sum(same_valid) == 0.0, "全同图批未正确屏蔽负例！"

    # 3c. 验证公开榜 MMStar / Hallusion / MathVista 的 3 次独立随机图像组异图置换（seed 101, 102, 103）
    import hashlib
    pub_derangements = {}
    for bname in ("mmstar", "hallusion", "mathvista"):
        items = bd.load_bench(bname, 0)
        perms = {}
        for s_seed in (101, 102, 103):
            perm, same_cnt = eb.build_derangement_by_image(items, shift_seed=s_seed)
            assert same_cnt == 0, f"{bname} seed={s_seed} 存在同图碰撞"
            perms[s_seed] = hashlib.sha1(json.dumps(perm).encode()).hexdigest()
        assert len(set(perms.values())) == 3, f"{bname} 三次异图置换哈希重复！"
        pub_derangements[bname] = perms

    report["checks"]["3_derangements"] = {
        "passed": True,
        "internal_val": internal_derangements,
        "inbatch_negatives_checked_batches": inbatch_checked_batches,
        "inbatch_negatives_valid_pairs": inbatch_valid_pairs,
        "inbatch_all_same_masked": True,
        "public_benchmarks_3seeds": pub_derangements,
    }
    report["checks"]["4_unified_entrypoint_split"] = {
        "passed": True,
        "front_joint_final_exact_match": True,
        "split_info": split_info,
        "val_1000_len": len(val_1000),
        "val_1000_sha1": val_1000_sha1,
    }
    print("[PASS] 3 & 4. Cross-stage entrypoint splits & internal/in-batch/public derangements verified.")

    # 5 & 6. 验证 A1 (CA-only) 前向/梯度生效、A3+B2 冻结与文本旁路逐位一致
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = f"{B}/zk/model/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()

    test_ids = tok("Question: What is 12 + 35? Answer:", return_tensors="pt")["input_ids"].to(dev)
    test_mask = torch.ones_like(test_ids)
    with torch.no_grad():
        logits_orig = model(input_ids=test_ids, attention_mask=test_mask).logits.detach().clone()

    pre_hashes, pre_bb_sha1 = prov.snapshot_backbone_hashes(model)

    # 5a. 完整验证 A1 (no_rot=True, hybrid=True, direct_hyb=True)：零旋转参数、有梯度、更新后改变有图输出、无图逐位一致
    layers = [0, 4, 8, 12, 16, 20, 24]
    vrmgs_a1, reg_a1 = attach_rotation(
        model, layers, block_size=64, visual_dim=4096, query_dim=256,
        multifreq=False, hybrid=True, qcond=False, direct_hyb=True, no_rot=True
    )
    assert all(m.queries is None for m in vrmgs_a1), "A1 (no_rot=True) 不应创建 queries 旋转参数！"
    a1_params = [p for m in vrmgs_a1 for p in m.parameters()]
    freeze_all_but(model, a1_params)
    opt_a1 = torch.optim.AdamW(a1_params, lr=1e-3)
    prov.verify_frozen_backbone_and_optimizer(model, opt_a1, a1_params)

    vis_dummy = torch.randn(2, 16, 4096, device=dev)
    vm_dummy = torch.ones(2, 16, device=dev)
    ids_b = test_ids.expand(2, -1)
    mask_b = test_mask.expand(2, -1)
    for step_a1 in range(2):
        opt_a1.zero_grad(set_to_none=True)
        with rotation_context_with(reg_a1, vis_dummy, vm_dummy, ans_start=[3, 3], text_mask=mask_b):
            lg_a1 = model(input_ids=ids_b, attention_mask=mask_b).logits.float()
        loss_a1 = lg_a1.pow(2).mean()
        loss_a1.backward()
        if step_a1 == 1:
            a1_grad_norm = sum(p.grad.detach().float().norm().item() for p in a1_params if p.grad is not None)
            assert a1_grad_norm > 0.0, "A1 视觉 cross-attention 未获得非零梯度！"
        opt_a1.step()

    with torch.no_grad():
        with rotation_context_with(reg_a1, vis_dummy[:1], vm_dummy[:1], ans_start=[3], text_mask=test_mask):
            logits_a1_vis = model(input_ids=test_ids, attention_mask=test_mask).logits.detach().clone()
        logits_a1_no_img = model(input_ids=test_ids, attention_mask=test_mask).logits.detach().clone()
    a1_vis_effect = (logits_orig.float() - logits_a1_vis.float()).abs().max().item()
    a1_no_img_diff = (logits_orig.float() - logits_a1_no_img.float()).abs().max().item()
    assert a1_vis_effect > 1e-4, f"A1 更新后有图前向未改变输出: diff={a1_vis_effect}"
    assert a1_no_img_diff == 0.0 and torch.equal(logits_orig, logits_a1_no_img), "A1 无图前向与原模型不一致！"
    prov.verify_frozen_backbone_and_optimizer(model, opt_a1, a1_params, pre_hashes)

    # 5b & 6. 重新加载干净模型测 A3 + PatchFront (rots K=4) 的训练冻结与旁路一致性
    del model, vrmgs_a1, reg_a1, opt_a1
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=dev,
                                                 trust_remote_code=True).eval()
    vrmgs, registry = attach_rotation(
        model, layers, block_size=64, visual_dim=4096, query_dim=256,
        wrap_extra=[1, 2, 3], multifreq=True, hybrid=True, qcond=True, direct_hyb=True, no_rot=False
    )
    front = PatchFront(model, K=4, mode="rots", vdim=4096).to(dev)
    fparams = front.trainable_params()
    vparams = [p for m in vrmgs for p in m.parameters()]
    freeze_all_but(model, [])
    for p in fparams + vparams:
        p.requires_grad_(True)

    opt = torch.optim.AdamW([{"params": vparams, "lr": 1e-3}, {"params": fparams, "lr": 1e-3}])
    prov.verify_frozen_backbone_and_optimizer(model, opt, fparams + vparams, pre_hashes)

    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        pv_dummy = torch.randn(2, 64, 1536, device=dev, dtype=torch.float16)
        grid_dummy = torch.tensor([[1, 8, 8], [1, 8, 8]], device=dev)
        v_hat, cl = front(pv_dummy, grid_dummy, [64, 64])
        vm = torch.ones(2, 16, device=dev)
        with rotation_context_with(registry, v_hat, vm, ans_start=[3, 3], text_mask=mask_b):
            lg = model(input_ids=ids_b, attention_mask=mask_b).logits.float()
        loss = lg.pow(2).mean()
        loss.backward()
        opt.step()

    frozen_res = prov.verify_frozen_backbone_and_optimizer(model, opt, fparams + vparams, pre_hashes)

    front.eval()
    for m in vrmgs:
        m.eval()
    with torch.no_grad():
        logits_no_img_1 = model(input_ids=test_ids, attention_mask=test_mask).logits.detach().clone()
        pv_1 = torch.randn(1, 64, 1536, device=dev, dtype=torch.float16)
        grid_1 = torch.tensor([[1, 8, 8]], device=dev)
        vh_1, _ = front(pv_1, grid_1, [64])
        vm_1 = torch.ones(1, 16, device=dev)
        with rotation_context_with(registry, vh_1, vm_1, ans_start=[3], text_mask=test_mask):
            logits_with_img = model(input_ids=test_ids, attention_mask=test_mask).logits.detach().clone()
        logits_no_img_2 = model(input_ids=test_ids, attention_mask=test_mask).logits.detach().clone()

    max_diff_1 = (logits_orig.float() - logits_no_img_1.float()).abs().max().item()
    max_diff_2 = (logits_orig.float() - logits_no_img_2.float()).abs().max().item()
    vis_effect = (logits_orig.float() - logits_with_img.float()).abs().max().item()
    assert max_diff_1 == 0.0 and torch.equal(logits_orig, logits_no_img_1)
    assert max_diff_2 == 0.0 and torch.equal(logits_orig, logits_no_img_2)
    assert vis_effect > 1e-4

    report["checks"]["5_frozen_backbone_and_a1"] = {
        "passed": True,
        **frozen_res,
        "a1_ca_only_vrmg_params": sum(p.numel() for p in a1_params),
        "a1_grad_norm": a1_grad_norm,
        "a1_vis_effect": a1_vis_effect,
        "a1_no_img_diff": a1_no_img_diff,
        "a3_full_vrmg_params": int(sum(p.numel() for p in vparams)),
        "b2_rots_front_total_params": int(sum(p.numel() for p in fparams)),
        "b2_rots_static_only_params": int(sum(p.numel() for p in front.static.parameters())),
    }
    report["checks"]["6_text_bypass_bit_exact"] = {
        "passed": True,
        "max_abs_diff_before_vision": max_diff_1,
        "max_abs_diff_after_vision": max_diff_2,
        "max_abs_diff_with_vision": vis_effect,
        "bit_equal_before_and_after": True,
    }
    print(f"[PASS] 5 & 6. A1 CA-only (grad={a1_grad_norm:.4f}, vis_diff={a1_vis_effect:.4f}), Frozen backbone (311/311 bit-exact), & text bypass verified.")

    # 7. 记录完整 manifest
    report["manifest"] = prov.build_run_manifest("foundation_gate", gate_args, split_info,
                                                 extra={"val_1000_sha1": val_1000_sha1, "backbone_sha1": pre_bb_sha1})
    report["elapsed_seconds"] = time.time() - t0
    report["all_passed"] = True

    out_path = OUT_DIR / "foundation_gate_manifest.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ALL_7_FOUNDATION_GATES_PASSED in {report['elapsed_seconds']:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
