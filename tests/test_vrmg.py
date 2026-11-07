"""新版 VRMG 单测：正交性、激活等价、梯度、逐样本差异、零初始化恒等。"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrmg import LayerVRMG, RotatedLinear  # noqa: E402

BLOCK = 64
PROJS = {"q_proj": 256, "k_proj": 128, "down_proj": 512}
VIS = 96
B, L, T = 3, 11, 7


def log(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def main():
    torch.manual_seed(0)
    res = []
    vrmg = LayerVRMG(PROJS, block_size=BLOCK, visual_dim=VIS, query_dim=64, num_heads=4)
    torch.nn.init.normal_(vrmg.out.weight, std=0.02)
    visual = torch.randn(B, L, VIS)
    Rs = vrmg(visual)

    err = 0.0
    for name, R in Rs.items():
        eye = torch.eye(BLOCK).expand_as(R)
        err = max(err, ((R.transpose(-1, -2) @ R) - eye).abs().max().item())
    res.append(log("全部块正交 (RᵀR=I)", err < 1e-4, f"max_err={err:.2e}"))

    res.append(log("块数正确", Rs["q_proj"].shape[1] == 4 and Rs["k_proj"].shape[1] == 2
                   and Rs["down_proj"].shape[1] == 8,
                   f"{ {k: v.shape[1] for k, v in Rs.items()} }"))

    base = torch.nn.Linear(256, 40, bias=False)
    rot = RotatedLinear(base, "q_proj", BLOCK)
    x = torch.randn(B, T, 256)
    y = rot(x, Rs["q_proj"])
    y_ref = torch.stack([
        F.linear(x[j] @ torch.block_diag(*[Rs["q_proj"][j, i] for i in range(Rs["q_proj"].shape[1])]),
                 base.weight)
        for j in range(B)])
    d = (y - y_ref).abs().max().item()
    res.append(log("激活等价 y=W(Rᵀx)", d < 1e-4, f"max_diff={d:.2e}"))

    y.square().mean().backward()
    res.append(log("VRMG 收到梯度", vrmg.out.weight.grad is not None and vrmg.out.weight.grad.abs().sum() > 0))
    res.append(log("基础权重冻结", base.weight.grad is None))

    v2 = visual.clone()
    v2[0] += 1.0
    R2 = vrmg(v2)
    d01 = (R2["q_proj"][0] - Rs["q_proj"][0]).abs().max().item()
    d11 = (R2["q_proj"][1] - Rs["q_proj"][1]).abs().max().item()
    res.append(log("逐样本旋转不同", d01 > 1e-3 and d11 < 1e-6, f"{d01:.2e} / {d11:.2e}"))

    fresh = LayerVRMG(PROJS, block_size=BLOCK, visual_dim=VIS, query_dim=64, num_heads=4)
    rot0 = RotatedLinear(base, "q_proj", BLOCK)
    d_id = (rot0(x, fresh(visual)["q_proj"]) - base(x)).abs().max().item()
    res.append(log("零初始化=R=I", d_id < 1e-6, f"max_diff={d_id:.2e}"))

    print(f"\n{sum(res)}/{len(res)} passed")
    return 0 if all(res) else 1


if __name__ == "__main__":
    raise SystemExit(main())
