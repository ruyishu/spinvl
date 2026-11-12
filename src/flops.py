"""Module/shape-based prefill FLOP estimate, separate from latency measurement.

Counts multiply/add operations in executed Linear/Conv/MHA, attention products,
rotation products, and an explicitly approximate LU+solve cost for Cayley.
Normalization, softmax, activations, indexing, and memory traffic are excluded.
This is an algorithmic estimate, not hardware-counter FLOPs.
"""
from collections import defaultdict
import math
import torch
from torch import nn
import wrap


class PrefillFlopEstimate:
    def __init__(self, roots):
        self.roots = roots
        self.handles = []
        self.counts = defaultdict(float)
        self.events = defaultdict(float)
        self.unknown_attention = set()

    def _add(self, category, name, value):
        self.counts[category] += float(value)
        self.events[f"{name}:{category}"] += float(value)

    def _hook(self, name, module, args, kw, out):
        x = args[0] if args else kw.get("hidden_states")
        if isinstance(module, nn.Linear):
            self._add("linear_matmul", name, 2*x.numel()*module.out_features)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            positions = out.numel()/module.out_channels
            self._add("convolution", name, 2*positions*module.weight.numel())
        elif isinstance(module, nn.MultiheadAttention):
            q, k, v = args[:3]
            batch, nq = (q.shape[0], q.shape[1]) if module.batch_first else (q.shape[1], q.shape[0])
            nk = k.shape[1] if module.batch_first else k.shape[0]
            nv = v.shape[1] if module.batch_first else v.shape[0]
            e = module.embed_dim
            # MHA uses functional projections; its Linear child hooks do not run.
            projections = 2*batch*(nq*q.shape[-1]*e+nk*k.shape[-1]*e+nv*v.shape[-1]*e+nq*e*e)
            self._add("mha_projections", name, projections)
            self._add("attention_products", name, 4*batch*nq*nk*e)
        elif all(hasattr(module, a) for a in ("q_proj", "k_proj", "v_proj", "o_proj")):
            # Text attention: exact projection widths, including GQA/head_dim != hidden_size.
            batch, length = x.shape[:2]
            q_width = module.q_proj.weight.shape[0]
            causal = bool(getattr(module, "is_causal", True)) and wrap._STATIC is None
            pairs = length*(length+1)/2 if causal else length*length
            self._add("attention_products", name, 4*batch*pairs*q_width)
        elif hasattr(module, "qkv") and hasattr(module, "proj") and "Attention" in type(module).__name__:
            cu = kw.get("cu_seqlens", args[1] if len(args)>1 else None)
            if cu is None:
                raise ValueError(f"Missing vision attention segment lengths: {name}")
            lengths = (cu[1:]-cu[:-1]).detach().cpu().tolist()
            width = module.qkv.out_features//3
            self._add("attention_products", name, 4*sum(n*n for n in lengths)*width)
        elif type(module).__name__ == "LayerVRMG":
            batch, cells = x.shape[:2]
            if not module.no_rot:
                b = module.block_size
                self._add("cayley_lu_solve_estimate", name, batch*module.total_blocks*(8.0/3)*b**3)
                if module.multifreq:
                    pairs = module.n_low+(module.n_mid+module.n_high)*cells
                    self._add("attention_products", name, 4*batch*pairs*module.qdim)
        elif type(module).__name__ == "PatchFront":
            static = module.static
            if static is not None and static.kind in ("rot", "rots"):
                blocks = sum(p.shape[0] for p in static.params.values())
                self._add("cayley_lu_solve_estimate", name, blocks*(8.0/3)*static.block**3)
        elif type(module).__name__ == "RotatedProj" and wrap._STATIC is not None:
            static = wrap._STATIC
            key = f"{module.layer_id}_{module.proj_name}"
            if static.kind in ("rot", "rots") and key in static.params:
                self._add("rotation_products", name, 2*x.numel()*static.block)
                if static.kind == "rots":
                    self._add("channel_scaling", name, x.numel())
        elif type(module).__name__ == "RotatedLinear":
            rotation = args[1] if len(args)>1 else kw.get("R")
            if rotation is not None:
                self._add("rotation_products", name, 2*x.numel()*module.block_size)
        elif "Attention" in type(module).__name__:
            self.unknown_attention.add(name+":"+type(module).__name__)

    def __enter__(self):
        seen = set()
        for prefix, root in self.roots.items():
            if root is None:
                continue
            for suffix, module in root.named_modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                name = prefix+("."+suffix if suffix else "")
                def hook(m, args, kw, out, name=name):
                    self._hook(name, m, args, kw, out)
                self.handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def result(self):
        if self.unknown_attention:
            raise ValueError(f"Unaccounted attention modules: {sorted(self.unknown_attention)}")
        return {"est_prefill_tflops": sum(self.counts.values())/1e12,
                "flops_method": "executed_module_shapes_plus_estimated_cayley_lu_v1",
                "flops_components": dict(self.counts), "flops_events": dict(self.events),
                "flops_scope": {"multiply_add": 2, "causal_attention_pairs": "L*(L+1)/2",
                    "bidirectional_attention_pairs": "L*L (segmented for vision)",
                    "cayley": "estimated LU factorization plus b right-hand sides: 8*b^3/3",
                    "excluded": ["bias additions", "normalizations", "softmax", "activations",
                                 "RoPE/positional functions", "gating elementwise ops", "memory traffic"],
                    "hardware_measured": False}}
