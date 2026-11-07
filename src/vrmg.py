"""ROSE 式视觉条件参数旋转 + SpinVL 问题条件化（qcond）与无损局部交叉寻址（direct_hyb）。

核心保持：**0 个视觉 Token 进入 LLM 主干自注意力**（省 74% FLOPs 不变）。
新增两项可选升级（默认关闭，100% 兼容全部旧权重）：
1. `qcond=True`（Question-Conditioned Rotation）：把当前层题目文本池化向量经零初始化投影叠到 `self.queries` 上再查图，
   让旋转矩阵 R_{v,q} 带着题目问题去提取对应区域特征；
2. `direct_hyb=True`（Uncompressed Direct Cross-Attn）：在 `hybrid` 的 256 维窄子空间里，去掉 16 锚点的有损池化，
   让每个文本词直接对全图所有视觉 cell 做轻量交叉注意力（带 LayerNorm 稳梯度），恢复图表数值与 OCR 单词的细粒度读取。
"""

import torch
import torch.nn as nn


class LayerVRMG(nn.Module):
    """一层一个：把视觉特征变成该层各投影的块对角正交矩阵。"""

    def __init__(self, proj_dims: dict, block_size: int = 64, visual_dim: int = 2560,
                 query_dim: int = 256, num_heads: int = 8, identity_init: bool = True,
                 multifreq: bool = False, tokrot: bool = False, hybrid: bool = False,
                 n_anchors: int = 16, prompt_alpha: float = 1.0, hyb_bound: float = 0.0,
                 qcond: bool = False, direct_hyb: bool = False):
        super().__init__()
        self.proj_dims = dict(proj_dims)
        self.block_size = block_size
        self.tri = block_size * (block_size - 1) // 2
        self.multifreq = multifreq
        self.tokrot = tokrot
        self.hybrid = hybrid
        self.prompt_alpha = float(prompt_alpha)
        self.hyb_bound = float(hyb_bound)
        self.qcond = bool(qcond)
        self.direct_hyb = bool(direct_hyb)
        self.d_model = proj_dims["q_proj"]
        self.qdim = query_dim

        self.blocks = {}
        for name, dim in self.proj_dims.items():
            assert dim % block_size == 0, f"{name}: {dim} not divisible by {block_size}"
            self.blocks[name] = dim // block_size

        offsets, total = {}, 0
        for name, nb in self.blocks.items():
            offsets[name] = total
            total += nb
        self.offsets = offsets
        self.total_blocks = total

        self.queries = nn.Parameter(torch.randn(total, query_dim) * 0.02)
        self.k_proj = nn.Linear(visual_dim, query_dim)
        self.v_proj = nn.Linear(visual_dim, query_dim)
        self.attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
        self.out = nn.Linear(query_dim, self.tri)

        if self.qcond:
            self.qc_norm = nn.LayerNorm(self.d_model)
            self.qc_proj = nn.Linear(self.d_model, query_dim)
            nn.init.zeros_(self.qc_proj.weight)
            nn.init.zeros_(self.qc_proj.bias)

        if multifreq:
            self.n_low = max(1, round(total * 0.25))
            self.n_high = max(1, round(total * 0.375))
            self.n_mid = total - self.n_low - self.n_high
            self.gate = nn.Linear(proj_dims["q_proj"], total)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

        if tokrot:
            nb_d = self.d_model // block_size
            self.nb_d = nb_d
            self.tok_norm = nn.LayerNorm(self.d_model)
            self.tok_q = nn.Linear(self.d_model, query_dim)
            self.tok_k = nn.Linear(visual_dim, query_dim)
            self.tok_v = nn.Linear(visual_dim, query_dim)
            self.tok_attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
            self.tok_projs = [n for n, d in self.proj_dims.items() if d == self.d_model]
            self.tok_idx = {n: i for i, n in enumerate(self.tok_projs)}
            n_tp = len(self.tok_projs)
            self.u1 = nn.Parameter(torch.randn(n_tp, nb_d, block_size))
            self.tok_theta = nn.Linear(query_dim, n_tp * nb_d * (block_size // 2))
            self.tok_delta = nn.Linear(query_dim, n_tp * nb_d * block_size)
            nn.init.zeros_(self.tok_theta.weight)
            nn.init.zeros_(self.tok_theta.bias)
            nn.init.zeros_(self.tok_delta.weight)
            nn.init.zeros_(self.tok_delta.bias)

        if hybrid:
            self.n_anchors = n_anchors
            if not self.direct_hyb:
                self.anc_q = nn.Parameter(torch.randn(n_anchors, query_dim) * 0.02)
                self.anc_pool = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
            else:
                self.hyb_post_norm = nn.LayerNorm(query_dim)
            self.anc_k = nn.Linear(visual_dim, query_dim)
            self.anc_v = nn.Linear(visual_dim, query_dim)
            self.hyb_norm = nn.LayerNorm(self.d_model)
            self.hyb_q = nn.Linear(self.d_model, query_dim)
            self.hyb_attn = nn.MultiheadAttention(query_dim, num_heads, batch_first=True)
            self.hyb_out = nn.Linear(query_dim, self.d_model)
            nn.init.zeros_(self.hyb_out.weight)
            nn.init.zeros_(self.hyb_out.bias)

        if identity_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    @staticmethod
    def _pos_encoding(L: int, dim: int, device, base: float):
        pos = torch.arange(L, device=device, dtype=torch.float32)
        freqs = base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        ang = pos[:, None] * freqs[None, :]
        pe = torch.zeros(L, dim, device=device)
        pe[:, 0::2] = torch.sin(ang)
        pe[:, 1::2] = torch.cos(ang)
        return pe

    def _band_attn(self, q, k, v, key_padding_mask, tau):
        score = (q @ k.transpose(-1, -2)) / (self.qdim ** 0.5 * tau)
        if key_padding_mask is not None:
            score = score.masked_fill(key_padding_mask[:, None, :], float("-inf"))
        w = torch.softmax(score.float(), dim=-1).to(v.dtype)
        return w @ v

    def forward(self, visual: torch.Tensor, visual_mask: torch.Tensor = None,
                x_text: torch.Tensor = None) -> dict:
        bsz = visual.shape[0]
        kpm = (visual_mask == 0) if visual_mask is not None else None
        qs = self.queries.unsqueeze(0).expand(bsz, -1, -1)
        if self.qcond and x_text is not None:
            if x_text.shape[0] == bsz:
                q_txt = self.qc_proj(self.qc_norm(x_text.float().mean(dim=1, keepdim=True))).to(qs.dtype)
                qs = qs + q_txt

        if not self.multifreq:
            k = self.k_proj(visual)
            v = self.v_proj(visual)
            ctx, _ = self.attn(qs, k, v, key_padding_mask=kpm, need_weights=False)
        else:
            L = visual.shape[1]
            dev = visual.device
            k_all, v_all = self.k_proj(visual), self.v_proj(visual)
            if visual_mask is not None:
                pooled = (visual * visual_mask.unsqueeze(-1)).sum(1, keepdim=True) / \
                         visual_mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
            else:
                pooled = visual.mean(1, keepdim=True)
            k_low, v_low = self.k_proj(pooled), self.v_proj(pooled)
            pe_mid = self._pos_encoding(L, self.qdim, dev, base=10000.0).to(k_all.dtype)
            pe_high = self._pos_encoding(L, self.qdim, dev, base=50.0).to(k_all.dtype)
            i0, i1 = self.n_low, self.n_low + self.n_mid
            ctxs = [
                self._band_attn(qs[:, :i0], k_low, v_low, None, 1.0),
                self._band_attn(qs[:, i0:i1], k_all + pe_mid, v_all, kpm, 1.0),
                self._band_attn(qs[:, i1:], k_all + pe_high, v_all, kpm, 0.25),
            ]
            ctx = torch.cat(ctxs, dim=1)
        vals = self.out(ctx)

        out = {}
        for name, nb in self.blocks.items():
            seg = vals[:, self.offsets[name] : self.offsets[name] + nb]
            out[name] = _cayley_blocks(seg, self.block_size)
        if self.multifreq:
            out["gate"] = (self.gate, self.offsets)
        if self.tokrot:
            pe = self._pos_encoding(visual.shape[1], self.qdim, visual.device, base=500.0).to(visual.dtype)
            tk = self.tok_k(visual) + pe
            tv = self.tok_v(visual)
            out["tokrot"] = (self, tk, tv, kpm, {})
        if self.hybrid:
            pe = self._pos_encoding(visual.shape[1], self.qdim, visual.device, base=500.0).to(visual.dtype)
            ak = self.anc_k(visual) + pe
            av = self.anc_v(visual)
            if self.direct_hyb:
                out["hybrid"] = (self, (ak, av, kpm), {})
            else:
                aq = self.anc_q.unsqueeze(0).expand(bsz, -1, -1)
                anc, _ = self.anc_pool(aq, ak, av, key_padding_mask=kpm, need_weights=False)
                out["hybrid"] = (self, anc, {})
        if self.prompt_alpha != 1.0 or self.hyb_bound > 0.0:
            out["cfg"] = (self.prompt_alpha, self.hyb_bound)
        return out

    def apply_tokrot(self, xr: torch.Tensor, x_raw: torch.Tensor, proj_name: str,
                     tk: torch.Tensor, tv: torch.Tensor, kpm, tcache: dict) -> torch.Tensor:
        if proj_name not in self.tok_idx:
            return xr
        bsz, seq, nb, b = xr.shape
        ckey = (x_raw.data_ptr(), x_raw.shape)
        if ckey not in tcache:
            q_t = self.tok_q(self.tok_norm(x_raw.float()))
            tk_use = tk.expand(bsz, -1, -1) if tk.shape[0] == 1 and bsz > 1 else tk
            tv_use = tv.expand(bsz, -1, -1) if tv.shape[0] == 1 and bsz > 1 else tv
            kpm_use = kpm.expand(bsz, -1) if (kpm is not None and kpm.shape[0] == 1 and bsz > 1) else kpm
            ctx_t, _ = self.tok_attn(q_t, tk_use, tv_use, key_padding_mask=kpm_use, need_weights=False)
            n_tp = len(self.tok_projs)
            thetas = self.tok_theta(ctx_t).reshape(bsz, seq, n_tp, nb, b // 2)
            deltas = self.tok_delta(ctx_t).reshape(bsz, seq, n_tp, nb, b)
            tcache[ckey] = (thetas, deltas)
        thetas, deltas = tcache[ckey]
        pi = self.tok_idx[proj_name]
        th = thetas[:, :, pi]
        dl = deltas[:, :, pi]
        c, s = torch.cos(th).to(xr.dtype), torch.sin(th).to(xr.dtype)
        x0, x1 = xr[..., 0::2], xr[..., 1::2]
        y = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)
        u1 = torch.nn.functional.normalize(self.u1[pi].float(), dim=-1)
        u2 = torch.nn.functional.normalize(u1.unsqueeze(0).unsqueeze(0) + dl.float(), dim=-1)
        u1_d, u2_d = u1.to(y.dtype), u2.to(y.dtype)
        y = y - 2.0 * (y * u1_d).sum(dim=-1, keepdim=True) * u1_d
        y = y - 2.0 * (y * u2_d).sum(dim=-1, keepdim=True) * u2_d
        return y

    def apply_hybrid(self, x_raw: torch.Tensor, anc, hcache: dict) -> torch.Tensor:
        bsz, seq, dim = x_raw.shape
        if dim != self.d_model:
            return None
        ckey = id(x_raw)
        hit = hcache.get(ckey)
        if hit is None or hit[0] is not x_raw:
            q_t = self.hyb_q(self.hyb_norm(x_raw.float()))
            if self.direct_hyb:
                ak, av, kpm = anc
                ak_u = ak.expand(bsz, -1, -1) if ak.shape[0] == 1 and bsz > 1 else ak
                av_u = av.expand(bsz, -1, -1) if av.shape[0] == 1 and bsz > 1 else av
                kpm_u = kpm.expand(bsz, -1) if (kpm is not None and kpm.shape[0] == 1 and bsz > 1) else kpm
                ctx_t, _ = self.hyb_attn(q_t, ak_u, av_u, key_padding_mask=kpm_u, need_weights=False)
                ctx_t = self.hyb_post_norm(ctx_t)
            else:
                anc_use = anc.expand(bsz, -1, -1) if anc.shape[0] == 1 and bsz > 1 else anc
                ctx_t, _ = self.hyb_attn(q_t, anc_use, anc_use, need_weights=False)
            dx = self.hyb_out(ctx_t)
            if self.hyb_bound > 0.0:
                xn = x_raw.float().norm(dim=-1, keepdim=True).clamp(min=1e-4)
                dn = dx.norm(dim=-1, keepdim=True).clamp(min=1e-4)
                dx = dx * torch.clamp(self.hyb_bound * xn / dn, max=1.0)
            hit = (x_raw, dx.to(x_raw.dtype))
            hcache[ckey] = hit
        return hit[1]


def _cayley_blocks(vals: torch.Tensor, b: int) -> torch.Tensor:
    bsz, nb, _ = vals.shape
    idx = torch.tril_indices(b, b, offset=-1, device=vals.device)
    P = torch.zeros(bsz, nb, b, b, device=vals.device, dtype=vals.dtype)
    P[:, :, idx[0], idx[1]] = vals
    P = P - P.transpose(-1, -2)
    eye = torch.eye(b, device=vals.device, dtype=vals.dtype)
    A = (eye - P).transpose(-1, -2)
    Bm = (eye + P).transpose(-1, -2)
    return torch.linalg.solve(A, Bm).transpose(-1, -2)


class RotatedLinear(nn.Module):
    def __init__(self, base: nn.Linear, proj_name: str, block_size: int):
        super().__init__()
        self.base = base
        self.proj_name = proj_name
        self.block_size = block_size
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, R: torch.Tensor, gate=None, tokrot=None, hybrid=None, cfg=None) -> torch.Tensor:
        bsz, seq, dim = x.shape
        nb = R.shape[1]
        if R.shape[0] == 1 and bsz > 1:
            R = R.expand(bsz, -1, -1, -1)
        xb = x.reshape(bsz, seq, nb, self.block_size)
        xr = torch.einsum("btrk,brkj->btrj", xb, R.to(x.dtype))
        if gate is not None:
            gate_mod, offsets = gate
            if x.shape[-1] == gate_mod.in_features:
                g = (2 * torch.sigmoid(gate_mod(x.float()))).to(x.dtype)
                off = offsets[self.proj_name]
                g = g[:, :, off : off + nb].unsqueeze(-1)
                xr = xb + g * (xr - xb)
        if tokrot is not None:
            vrmg, tk, tv, kpm, tcache = tokrot
            xr = vrmg.apply_tokrot(xr, x, self.proj_name, tk, tv, kpm, tcache)
        out_in = xr.reshape(bsz, seq, dim)
        if hybrid is not None:
            vrmg, anc, hcache = hybrid
            dx = vrmg.apply_hybrid(x, anc, hcache)
            if dx is not None:
                out_in = out_in + dx
        return self.base(out_in)
