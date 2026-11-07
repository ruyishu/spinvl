"""把旋转挂到 HF LLM 上：每 layer 一个 LayerVRMG，前向时每层只算一次。"""

from contextlib import contextmanager

import torch
import torch.nn as nn

from vrmg import LayerVRMG, RotatedLinear

_CTX = None
_STATIC = None

ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


def set_static(module):
    global _STATIC
    _STATIC = module


@contextmanager
def rotation_context_with(registry, visual, visual_mask=None):
    global _CTX
    prev = _CTX
    _CTX = {"visual": visual, "mask": visual_mask, "cache": {}, "modules": registry}
    try:
        yield
    finally:
        _CTX = prev


class RotatedProj(nn.Module):
    def __init__(self, base: nn.Linear, layer_id: int, proj_name: str, block_size: int):
        super().__init__()
        self.rot = RotatedLinear(base, proj_name, block_size)
        self.layer_id = layer_id
        self.proj_name = proj_name

    @property
    def weight(self):
        return self.rot.base.weight

    @property
    def bias(self):
        return self.rot.base.bias

    def forward(self, x):
        if _STATIC is not None:
            return _STATIC.apply(self.layer_id, self.proj_name, self.rot.base, x)
        if _CTX is None or self.layer_id not in _CTX["modules"]:
            return self.rot.base(x)
        cache = _CTX["cache"]
        if self.layer_id not in cache:
            cache[self.layer_id] = _CTX["modules"][self.layer_id](_CTX["visual"], _CTX["mask"], x)
        entry = cache[self.layer_id]
        return self.rot(x, entry[self.proj_name], entry.get("gate"), entry.get("tokrot"),
                        entry.get("hybrid"), entry.get("cfg"))


def _wrap_layer(layer, li, block_size):
    for parent, names in ((layer.self_attn, ATTN_PROJS), (layer.mlp, MLP_PROJS)):
        for n in names:
            base = getattr(parent, n)
            if not isinstance(base, RotatedProj):
                setattr(parent, n, RotatedProj(base, li, n, block_size))


def _in_features(m):
    return m.rot.base.in_features if isinstance(m, RotatedProj) else m.in_features


def attach_rotation(model: nn.Module, layer_indices, block_size: int = 64, visual_dim: int = 2560,
                    query_dim: int = 256, num_heads: int = 8, wrap_extra=(), multifreq: bool = False,
                    tokrot: bool = False, hybrid: bool = False, prompt_alpha: float = 1.0,
                    hyb_bound: float = 0.0, qcond: bool = False, direct_hyb: bool = False):
    layers = model.model.layers
    modules, registry = [], {}
    for li in layer_indices:
        layer = layers[li]
        proj_dims = {n: _in_features(getattr(layer.self_attn, n)) for n in ATTN_PROJS}
        proj_dims.update({n: _in_features(getattr(layer.mlp, n)) for n in MLP_PROJS})
        dev = getattr(layer.self_attn, "q_proj").weight.device
        vrmg = LayerVRMG(proj_dims, block_size=block_size, visual_dim=visual_dim,
                         query_dim=query_dim, num_heads=num_heads, multifreq=multifreq,
                         tokrot=tokrot, hybrid=hybrid, prompt_alpha=prompt_alpha,
                         hyb_bound=hyb_bound, qcond=qcond, direct_hyb=direct_hyb).to(dev, dtype=torch.float32)
        registry[li] = vrmg
        modules.append(vrmg)
        _wrap_layer(layer, li, block_size)
    for li in wrap_extra:
        _wrap_layer(layers[li], li, block_size)
    return modules, registry


def freeze_all_but(model: nn.Module, keep_params):
    keep_ids = {id(p) for p in keep_params}
    for p in model.parameters():
        p.requires_grad_(id(p) in keep_ids)
    return sum(p.numel() for p in keep_params)
