"""叙事 2：把 LLM 前 K 层改造成「内置视觉编码器」（PatchFront）。

数据流（只在「读图」阶段发生）：
    teacher 同款 image_processor 切出的 patch（已按 2×2 合并窗口排好序，4 个 patch = 1 个 cell）
      → Linear patch_embed + 行/列位置嵌入
      → 学生前 K 层（共享冻结权重 + 「静态正交旋转 / LoRA」或「整层复制可训」），双向注意力
      → 4 个 patch 拼接 → MLP merger → v̂ (n_cells, vdim)，与老师视觉塔输出逐 cell 对齐
读完图后图片 token 全部丢弃，v̂ 只作为 VRMG 的条件输入（叙事 1）去旋转文本通路。
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from vrmg import _cayley_blocks
import wrap

PROJ_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
PROJ_MLP = ("gate_proj", "up_proj", "down_proj")
MAX_GRID = 128


class StaticRot(nn.Module):
    kind = "rot"

    def __init__(self, layer_projs: dict, block_size: int = 64):
        super().__init__()
        self.block = block_size
        self.tri = block_size * (block_size - 1) // 2
        self.params = nn.ParameterDict()
        self.index = []
        for li, projs in layer_projs.items():
            for name, in_dim in projs.items():
                key = f"{li}_{name}"
                self.params[key] = nn.Parameter(torch.zeros(in_dim // block_size, self.tri))
                self.index.append((li, name, key))
        self._R = None

    def prepare(self):
        self._R = {}
        for li, name, key in self.index:
            self._R[(li, name)] = _cayley_blocks(self.params[key].float().unsqueeze(0), self.block)

    def apply(self, li, name, base, x):
        R = None if self._R is None else self._R.get((li, name))
        if R is None:
            return base(x)
        bsz, seq, dim = x.shape
        nb = R.shape[1]
        xb = x.reshape(bsz, seq, nb, self.block)
        xr = torch.einsum("btrk,rkj->btrj", xb, R[0].to(x.dtype)).reshape(bsz, seq, dim)
        return base(xr)


class StaticRotScale(StaticRot):
    kind = "rots"

    def __init__(self, layer_projs: dict, block_size: int = 64):
        super().__init__(layer_projs, block_size)
        self.scale = nn.ParameterDict()
        for li, projs in layer_projs.items():
            for name, in_dim in projs.items():
                self.scale[f"{li}_{name}"] = nn.Parameter(torch.ones(in_dim))

    def apply(self, li, name, base, x):
        R = None if self._R is None else self._R.get((li, name))
        d = self.scale.get(f"{li}_{name}")
        if R is None:
            return base(x)
        bsz, seq, dim = x.shape
        nb = R.shape[1]
        xb = x.reshape(bsz, seq, nb, self.block)
        xr = torch.einsum("btrk,rkj->btrj", xb, R[0].to(x.dtype)).reshape(bsz, seq, dim)
        return base(xr * d.to(x.dtype))


class StaticLoRA(nn.Module):
    kind = "lora"

    def __init__(self, layer_shapes: dict, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.scale = alpha / rank
        self.A = nn.ParameterDict()
        self.Bm = nn.ParameterDict()
        self.index = []
        for li, shapes in layer_shapes.items():
            for name, (din, dout) in shapes.items():
                key = f"{li}_{name}"
                self.A[key] = nn.Parameter(torch.randn(din, rank) * (1.0 / din ** 0.5))
                self.Bm[key] = nn.Parameter(torch.zeros(rank, dout))
                self.index.append((li, name, key))
        self._keys = {(li, n): k for li, n, k in self.index}

    def prepare(self):
        pass

    def apply(self, li, name, base, x):
        key = self._keys.get((li, name))
        out = base(x)
        if key is None:
            return out
        return out + ((x.to(self.A[key].dtype) @ self.A[key]) @ self.Bm[key] * self.scale).to(out.dtype)


def unwrap_layer(layer):
    layer = copy.deepcopy(layer)
    for parent, names in ((layer.self_attn, PROJ_ATTN), (layer.mlp, PROJ_MLP)):
        for n in names:
            m = getattr(parent, n)
            if hasattr(m, "rot"):
                setattr(parent, n, m.rot.base)
    return layer


class PatchFront(nn.Module):
    def __init__(self, model, K=4, mode="rot", vdim=4096, patch_dim=1536, block=64, lora_rank=16,
                 max_grid=MAX_GRID, merge_hidden=None):
        super().__init__()
        self.model = [model]
        self.K, self.mode, self.vdim = K, mode, vdim
        d = model.config.hidden_size
        self.d = d
        self.patch_embed = nn.Linear(patch_dim, d)
        self.row_emb = nn.Embedding(max_grid, d)
        self.col_emb = nn.Embedding(max_grid, d)
        nn.init.normal_(self.row_emb.weight, std=0.02)
        nn.init.normal_(self.col_emb.weight, std=0.02)
        mh = merge_hidden or vdim
        self.merger = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, mh), nn.GELU(), nn.Linear(mh, vdim))
        self.static = None
        self.copies = None
        layers = model.model.layers
        if K > 0 and mode in ("rot", "rots"):
            lp = {}
            for li in range(K):
                l = layers[li]
                lp[li] = {n: getattr(l.self_attn, n).rot.base.in_features for n in PROJ_ATTN}
                lp[li].update({n: getattr(l.mlp, n).rot.base.in_features for n in PROJ_MLP})
            self.static = (StaticRotScale if mode == "rots" else StaticRot)(lp, block)
        elif K > 0 and mode == "lora":
            ls = {}
            for li in range(K):
                l = layers[li]
                ls[li] = {n: (getattr(l.self_attn, n).rot.base.in_features,
                              getattr(l.self_attn, n).rot.base.out_features) for n in PROJ_ATTN}
                ls[li].update({n: (getattr(l.mlp, n).rot.base.in_features,
                                   getattr(l.mlp, n).rot.base.out_features) for n in PROJ_MLP})
            self.static = StaticLoRA(ls, lora_rank)
        elif K > 0 and mode == "full":
            self.copies = nn.ModuleList([unwrap_layer(layers[i]).float() for i in range(K)])

    def trainable_params(self):
        ps = list(self.patch_embed.parameters()) + list(self.row_emb.parameters()) + \
             list(self.col_emb.parameters()) + list(self.merger.parameters())
        if self.static is not None:
            ps += list(self.static.parameters())
        if self.copies is not None:
            ps += list(self.copies.parameters())
        return ps

    @staticmethod
    def rows_cols(grid_thw, device):
        rows, cols = [], []
        for t, h, w in grid_thw.tolist():
            hc, wc = h // 2, w // 2
            idx = torch.arange(hc * wc * 4, device=device)
            cell, sub = idx // 4, idx % 4
            ci, cj = cell // wc, cell % wc
            rows.append((2 * ci + sub // 2).clamp(max=MAX_GRID - 1))
            cols.append((2 * cj + sub % 2).clamp(max=MAX_GRID - 1))
        return rows, cols

    def forward(self, pixel_values, grid_thw, lens):
        dev = pixel_values.device
        B, L, _ = pixel_values.shape
        x = self.patch_embed(pixel_values.float())
        rows, cols = self.rows_cols(grid_thw, dev)
        pos = torch.zeros(B, L, self.d, device=dev)
        for i in range(B):
            n = int(lens[i])
            pos[i, :n] = self.row_emb(rows[i]) + self.col_emb(cols[i])
        x = x + pos
        valid = torch.arange(L, device=dev)[None, :] < torch.as_tensor(lens, device=dev)[:, None]

        model = self.model[0]
        if self.K > 0:
            neg = torch.finfo(torch.bfloat16).min
            mask = torch.zeros(B, 1, L, L, device=dev, dtype=torch.bfloat16)
            mask.masked_fill_(~valid[:, None, None, :], neg)
            position_ids = torch.arange(L, device=dev)[None, :].expand(B, -1)
            h = x
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if self.mode == "full":
                    cos, sin = model.model.rotary_emb(h.to(torch.bfloat16), position_ids)
                    for layer in self.copies:
                        o = layer(h, attention_mask=mask, position_ids=position_ids,
                                  position_embeddings=(cos, sin))
                        h = o[0] if isinstance(o, tuple) else o
                else:
                    if self.static is not None:
                        self.static.prepare()
                    cos, sin = model.model.rotary_emb(h.to(torch.bfloat16), position_ids)
                    wrap.set_static(self.static)
                    try:
                        for li in range(self.K):
                            o = model.model.layers[li](h.to(torch.bfloat16), attention_mask=mask,
                                                       position_ids=position_ids, position_embeddings=(cos, sin))
                            h = o[0] if isinstance(o, tuple) else o
                    finally:
                        wrap.set_static(None)
            x = h.float()

        Cmax = L // 4
        cells = x.reshape(B, Cmax, 4 * self.d)
        v_hat = self.merger(cells)
        cell_lens = [int(n) // 4 for n in lens]
        return v_hat, cell_lens


def feature_loss(v_hat, target_list, cell_lens, l1_weight=0.5):
    cos_terms, l1_terms = [], []
    for i, tgt in enumerate(target_list):
        n = cell_lens[i]
        assert tgt.shape[0] == n, f"cell 数不一致：pred {n} vs teacher {tgt.shape[0]}"
        p = v_hat[i, :n]
        t = tgt.to(p.device).float()
        cos_terms.append(1.0 - F.cosine_similarity(p, t, dim=-1))
        s = t.std().clamp(min=1e-3)
        l1_terms.append(F.smooth_l1_loss(p / s, t / s, reduction="none").mean(-1))
    cos = torch.cat(cos_terms)
    l1 = torch.cat(l1_terms)
    return cos.mean() + l1_weight * l1.mean(), (1.0 - cos).mean().detach()
