"""Step 1 冒烟：Qwen3-1.7B + 前 N 层 VRMG 旋转，验证接线、显存与步速。

跑法：
  python scripts/smoke_qwen.py --model /data/.../zk/model/Qwen3-1.7B --layers 0,1,2,3
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wrap import attach_rotation, freeze_all_but, rotation_context  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/ibu-bi-ai-share-disk/zk/model/Qwen3-1.7B")
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--query-dim", type=int, default=128)
    ap.add_argument("--visual-dim", type=int, default=1152)
    ap.add_argument("--visual-len", type=int, default=196)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = "cuda"
    dtype = torch.bfloat16

    print(f"== load {args.model} ==")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=dev, trust_remote_code=True)
    model.eval()
    n_layers = len(model.model.layers)
    print(f"layers={n_layers} hidden={model.config.hidden_size}")

    ids = torch.randint(100, 1000, (args.bs, args.seq), device=dev)
    labels = ids.clone()

    with torch.no_grad():
        base_logits = model(input_ids=ids).logits.float()

    def ce(logits):
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1)
        )

    base_loss = ce(base_logits).item()
    print(f"baseline loss = {base_loss:.4f}")

    layers = [int(x) for x in args.layers.split(",") if x != ""]
    vrmgs = attach_rotation(
        model, layers, num_blocks=args.blocks, visual_dim=args.visual_dim, query_dim=args.query_dim
    )
    vrmg_params = [p for m in vrmgs for p in m.parameters()]
    n_train = freeze_all_but(model, vrmg_params)
    print(f"wrapped layers={layers} vrmg_modules={len(vrmgs)} trainable={n_train/1e6:.1f}M")

    visual = torch.randn(args.bs, args.visual_len, args.visual_dim, device=dev, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        with rotation_context(visual):
            logits_id = model(input_ids=ids).logits.float()
    d_id = (logits_id - base_logits).abs().max().item()
    print(f"identity-init max|Δlogits| = {d_id:.3e}  (应为 0，证明接线无副作用)")

    for m in vrmgs:
        torch.nn.init.normal_(m.out.weight, std=0.02)
        torch.nn.init.normal_(m.out.bias, std=0.02)

    with torch.no_grad():
        with rotation_context(visual):
            logits_rot = model(input_ids=ids).logits.float()
    print(f"rotated loss = {ce(logits_rot).item():.4f} (与 baseline 的差 = {ce(logits_rot).item() - base_loss:+.4f})")

    opt = torch.optim.AdamW(vrmg_params, lr=1e-3)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.steps):
        with rotation_context(visual):
            out = model(input_ids=ids)
            loss = ce(out.logits.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(vrmg_params, 1.0)
        opt.step()
        if i == 0:
            print(f"step0 loss={loss.item():.4f} grad_norm={gnorm.item():.3e}")
    torch.cuda.synchronize()
    dt = (time.time() - t0) / args.steps
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"avg step = {dt*1000:.0f} ms | peak alloc = {peak:.1f} GiB | last loss = {loss.item():.4f}")

    any_nan = any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"grad contains nan: {any_nan}")
    print("SMOKE_OK" if (d_id < 1e-3 and not any_nan) else "SMOKE_FAIL")


if __name__ == "__main__":
    main()
